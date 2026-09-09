from fastapi import FastAPI, HTTPException, Request, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import json
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
import urllib3

from models import (
    Paper, PaperCreate, PaperUpdate, ArxivFetchRequest, ArxivFetchResponse, 
    FilterParams, AnalysisMetrics, ReadingStatus
)
from database import db

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="Papers Management API", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def extract_arxiv_id(url: str) -> Optional[str]:
    """Extract arXiv ID from URL or return the ID if directly provided."""
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)',
        r'(\d{4}\.\d{5,6})'  # Direct arXiv ID format
    ]
    
    for pattern in patterns:
        if match := re.search(pattern, url):
            return match.group(1)
    return None

def fetch_paper_details(arxiv_url: str) -> tuple[Optional[dict], Optional[str]]:
    """Fetch paper details from arXiv API."""
    try:
        arxiv_id = extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            return None, "Invalid arXiv URL format"
        
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        
        response = requests.get(api_url, verify=False, timeout=10)
        
        if response.status_code != 200:
            return None, f"API returned status code {response.status_code}"
        
        # XML namespaces for arXiv API
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        root = ET.fromstring(response.content)
        
        entry = root.find('.//atom:entry', ns)
        if entry is None:
            return None, f"No paper found with ID {arxiv_id}"
        
        title = entry.find('./atom:title', ns).text.strip()
        summary = entry.find('./atom:summary', ns).text.strip().replace('\n', ' ')
        
        published = entry.find('./atom:published', ns).text
        published_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
        
        # Extract categories/topics
        categories = []
        for category in entry.findall('./atom:category', ns):
            term = category.get('term')
            if term and '.' in term:
                primary = term.split('.')[0]
                if primary not in categories:
                    categories.append(primary)
        
        paper_details = {
            'title': title,
            'topics': categories,
            'description': summary,
            'date': published_date,
            'link': arxiv_url
        }
        
        return paper_details, None
        
    except requests.exceptions.RequestException as e:
        return None, f"Request error: {str(e)}"
    except Exception as e:
        return None, f"Error fetching paper details: {str(e)}"

def analyze_reading_habits(papers_df: pd.DataFrame) -> dict:
    """Generate analysis metrics for reading habits."""
    if papers_df.empty:
        return {
            "total_papers": 0,
            "read_papers": 0,
            "reading_papers": 0,
            "want_to_read_papers": 0,
            "reading_velocity": 0,
            "completion_rate": 0,
            "topic_distribution": {},
            "monthly_activity": {}
        }

    # Ensure date format is consistent
    papers_df['Date Added'] = pd.to_datetime(papers_df['Date Added'], errors='coerce')

    # Basic counts
    total_papers = len(papers_df)
    read_papers = len(papers_df[papers_df['Reading Status'] == 'Read'])
    reading_papers = len(papers_df[papers_df['Reading Status'] == 'Reading'])
    want_to_read_papers = len(papers_df[papers_df['Reading Status'] == 'Want to Read'])

    # Reading velocity (papers per month)
    if total_papers > 0:
        earliest_date = papers_df['Date Added'].min()
        latest_date = papers_df['Date Added'].max()

        if pd.notnull(earliest_date) and pd.notnull(latest_date):
            months_diff = (latest_date.year - earliest_date.year) * 12 + (latest_date.month - earliest_date.month)
            months_diff = max(1, months_diff)
            reading_velocity = total_papers / months_diff
        else:
            reading_velocity = 0
    else:
        reading_velocity = 0

    # Topic distribution
    topic_counts = Counter()
    for topics in papers_df['Topics']:
        if isinstance(topics, list):
            for topic in topics:
                topic_counts[topic] += 1

    topic_distribution = dict(topic_counts.most_common())

    # Monthly activity
    papers_df['Month'] = papers_df['Date Added'].dt.strftime('%Y-%m')
    monthly_activity = papers_df.groupby('Month').size().to_dict()

    # Completion rate
    completion_rate = (read_papers / total_papers * 100) if total_papers > 0 else 0

    return {
        "total_papers": total_papers,
        "read_papers": read_papers,
        "reading_papers": reading_papers,
        "want_to_read_papers": want_to_read_papers,
        "reading_velocity": round(reading_velocity, 2),
        "completion_rate": round(completion_rate, 1),
        "topic_distribution": topic_distribution,
        "monthly_activity": monthly_activity
    }

def create_status_chart(papers_df: pd.DataFrame) -> str:
    """Create status distribution chart."""
    if papers_df.empty:
        return ""
    
    status_counts = papers_df['Reading Status'].value_counts()
    
    fig = px.pie(
        names=status_counts.index,
        values=status_counts.values,
        color=status_counts.index,
        color_discrete_map={
            'Read': '#4CAF50',
            'Reading': '#FFC107', 
            'Want to Read': '#2196F3'
        },
        hole=0.4
    )
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_topic_chart(papers_df: pd.DataFrame) -> str:
    """Create topic distribution chart."""
    if papers_df.empty:
        return ""
    
    # Get topic counts
    topic_counts = Counter()
    for topics in papers_df['Topics']:
        if isinstance(topics, list):
            for topic in topics:
                topic_counts[topic] += 1
    
    if not topic_counts:
        return ""
    
    # Get top 10 topics
    top_topics = dict(topic_counts.most_common(10))
    
    fig = px.bar(
        x=list(top_topics.values()),
        y=list(top_topics.keys()),
        orientation='h',
        title="Top Research Topics"
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_monthly_activity_chart(papers_df: pd.DataFrame) -> str:
    """Create monthly activity chart."""
    if papers_df.empty:
        return ""
    
    papers_df['Date Added'] = pd.to_datetime(papers_df['Date Added'], errors='coerce')
    papers_df['Month'] = papers_df['Date Added'].dt.strftime('%Y-%m')
    monthly_counts = papers_df.groupby('Month').size()
    
    if monthly_counts.empty:
        return ""
    
    fig = px.line(
        x=monthly_counts.index,
        y=monthly_counts.values,
        title="Papers Added Over Time",
        labels={'x': 'Month', 'y': 'Number of Papers'}
    )
    fig.update_layout(height=300)
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# API Routes

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/papers", response_model=List[Paper])
async def get_papers():
    """Get all papers."""
    return db.get_all_papers()

@app.get("/api/papers/{paper_id}", response_model=Paper)
async def get_paper(paper_id: int):
    """Get a specific paper by ID."""
    paper = db.get_paper_by_id(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@app.post("/api/papers", response_model=Paper)
async def create_paper(paper: PaperCreate):
    """Create a new paper."""
    try:
        return db.create_paper(paper)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/api/papers/{paper_id}", response_model=Paper)
async def update_paper(paper_id: int, paper_update: PaperUpdate):
    """Update an existing paper."""
    paper = db.update_paper(paper_id, paper_update)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@app.delete("/api/papers/{paper_id}")
async def delete_paper(paper_id: int):
    """Delete a paper."""
    if not db.delete_paper(paper_id):
        raise HTTPException(status_code=404, detail="Paper not found")
    return {"message": "Paper deleted successfully"}

@app.post("/api/fetch-arxiv", response_model=ArxivFetchResponse)
async def fetch_arxiv_paper(request: ArxivFetchRequest):
    """Fetch paper details from arXiv."""
    paper_details, error = fetch_paper_details(request.arxiv_url)
    if error:
        raise HTTPException(status_code=400, detail=error)
    
    return ArxivFetchResponse(**paper_details)

@app.get("/api/topics")
async def get_topics():
    """Get all unique topics."""
    return {"topics": db.get_all_topics()}

@app.get("/api/analysis")
async def get_analysis():
    """Get reading analysis data."""
    papers_df = db.papers_df.copy()
    
    # Basic analysis
    analysis = analyze_reading_habits(papers_df)
    
    # Charts
    status_chart = create_status_chart(papers_df)
    topic_chart = create_topic_chart(papers_df)
    monthly_chart = create_monthly_activity_chart(papers_df)
    
    return {
        "analysis": analysis,
        "charts": {
            "status": status_chart,
            "topics": topic_chart,
            "monthly": monthly_chart
        }
    }

@app.get("/api/filter-papers")
async def filter_papers(
    status_filter: str = Query("All", description="Filter by reading status"),
    topic_filter: Optional[str] = Query(None, description="Comma-separated topics"),
    text_filter: Optional[str] = Query(None, description="Text search"),
    sort_by: str = Query("Date Added", description="Sort field"),
    sort_direction: str = Query("Descending", description="Sort direction")
):
    """Filter and sort papers."""
    papers_df = db.papers_df.copy()
    
    if papers_df.empty:
        return []
    
    # Apply status filter
    if status_filter != "All":
        papers_df = papers_df[papers_df['Reading Status'] == status_filter]
    
    # Apply topic filter
    if topic_filter:
        topics = [t.strip() for t in topic_filter.split(',')]
        papers_df = papers_df[
            papers_df['Topics'].apply(
                lambda x: any(topic in x for topic in topics) if isinstance(x, list) else False
            )
        ]
    
    # Apply text filter
    if text_filter:
        papers_df = papers_df[
            papers_df['Title'].str.contains(text_filter, case=False, na=False) |
            papers_df['Description'].str.contains(text_filter, case=False, na=False) |
            papers_df['Topics'].apply(
                lambda x: any(text_filter.lower() in t.lower() for t in x) if isinstance(x, list) else False
            )
        ]
    
    # Apply sorting
    ascending = sort_direction == "Ascending"
    if sort_by == "Date Added":
        papers_df['Date Added'] = pd.to_datetime(papers_df['Date Added'])
        papers_df = papers_df.sort_values('Date Added', ascending=ascending)
    elif sort_by == "Title":
        papers_df = papers_df.sort_values('Title', ascending=ascending)
    elif sort_by == "Reading Status":
        status_order = {"Read": 0, "Reading": 1, "Want to Read": 2}
        if ascending:
            status_order = {k: -v for k, v in status_order.items()}
        papers_df['Status Order'] = papers_df['Reading Status'].map(status_order)
        papers_df = papers_df.sort_values('Status Order', ascending=True)
    
    # Convert to Paper objects
    papers = []
    for idx, row in papers_df.iterrows():
        paper = Paper(
            id=idx,
            title=row['Title'],
            reading_status=row['Reading Status'],
            date_added=pd.to_datetime(row['Date Added']).date() if row['Date Added'] else None,
            link=row['Link'] if row['Link'] else None,
            topics=row['Topics'] if isinstance(row['Topics'], list) else [],
            description=row['Description'] if row['Description'] else None
        )
        papers.append(paper)
    
    return papers

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
