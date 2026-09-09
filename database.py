import json
import pandas as pd
from typing import List, Optional
from datetime import datetime
from models import Paper, PaperCreate, PaperUpdate, ReadingStatus
import os

PAPER_FILE = "paper_with_topics.json"

class PaperDatabase:
    def __init__(self):
        self.papers_df = self.load_papers()
    
    def load_papers(self) -> pd.DataFrame:
        """Load papers from JSON file"""
        if os.path.exists(PAPER_FILE):
            try:
                with open(PAPER_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data:
                    df = pd.DataFrame(data)
                    # Ensure all required columns exist
                    required_columns = ['Title', 'Reading Status', 'Date Added', 'Link', 'Topics', 'Description']
                    for col in required_columns:
                        if col not in df.columns:
                            if col == 'Topics':
                                df[col] = [[] for _ in range(len(df))]
                            else:
                                df[col] = ""
                    return df
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error loading papers: {e}")
        
        # Return empty DataFrame with required columns
        return pd.DataFrame(columns=['Title', 'Reading Status', 'Date Added', 'Link', 'Topics', 'Description'])
    
    def save_papers(self):
        """Save papers to JSON file"""
        try:
            # Convert DataFrame to list of dictionaries
            data = self.papers_df.to_dict('records')
            
            # Add Month field for each paper
            for paper in data:
                if paper.get('Date Added'):
                    try:
                        date_obj = pd.to_datetime(paper['Date Added'])
                        paper['Month'] = date_obj.strftime('%Y-%m')
                    except:
                        paper['Month'] = ""
            
            with open(PAPER_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Error saving papers: {e}")
    
    def get_all_papers(self) -> List[Paper]:
        """Get all papers"""
        papers = []
        for idx, row in self.papers_df.iterrows():
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
    
    def get_paper_by_id(self, paper_id: int) -> Optional[Paper]:
        """Get paper by ID"""
        if paper_id >= len(self.papers_df):
            return None
        
        row = self.papers_df.iloc[paper_id]
        return Paper(
            id=paper_id,
            title=row['Title'],
            reading_status=row['Reading Status'],
            date_added=pd.to_datetime(row['Date Added']).date() if row['Date Added'] else None,
            link=row['Link'] if row['Link'] else None,
            topics=row['Topics'] if isinstance(row['Topics'], list) else [],
            description=row['Description'] if row['Description'] else None
        )
    
    def create_paper(self, paper: PaperCreate) -> Paper:
        """Create a new paper"""
        # Check if paper with same title exists
        existing_titles = self.papers_df['Title'].str.lower()
        if paper.title.lower() in existing_titles.values:
            raise ValueError(f"A paper with title '{paper.title}' already exists!")
        
        # Create new paper row
        new_row = {
            'Title': paper.title,
            'Reading Status': paper.reading_status.value,
            'Date Added': paper.date_added.strftime('%Y-%m-%d'),
            'Link': paper.link or "",
            'Topics': paper.topics,
            'Description': paper.description or ""
        }
        
        # Add to DataFrame
        new_df = pd.DataFrame([new_row])
        self.papers_df = pd.concat([self.papers_df, new_df], ignore_index=True)
        
        # Sort by title
        self.papers_df = self.papers_df.sort_values('Title')
        self.papers_df = self.papers_df.reset_index(drop=True)
        
        # Save to file
        self.save_papers()
        
        # Return the created paper
        paper_id = self.papers_df[self.papers_df['Title'] == paper.title].index[0]
        return self.get_paper_by_id(paper_id)
    
    def update_paper(self, paper_id: int, paper_update: PaperUpdate) -> Optional[Paper]:
        """Update an existing paper"""
        if paper_id >= len(self.papers_df):
            return None
        
        # Update only provided fields
        if paper_update.title is not None:
            self.papers_df.at[paper_id, 'Title'] = paper_update.title
        if paper_update.reading_status is not None:
            self.papers_df.at[paper_id, 'Reading Status'] = paper_update.reading_status.value
        if paper_update.date_added is not None:
            self.papers_df.at[paper_id, 'Date Added'] = paper_update.date_added.strftime('%Y-%m-%d')
        if paper_update.link is not None:
            self.papers_df.at[paper_id, 'Link'] = paper_update.link
        if paper_update.topics is not None:
            self.papers_df.at[paper_id, 'Topics'] = paper_update.topics
        if paper_update.description is not None:
            self.papers_df.at[paper_id, 'Description'] = paper_update.description
        
        # Save to file
        self.save_papers()
        
        return self.get_paper_by_id(paper_id)
    
    def delete_paper(self, paper_id: int) -> bool:
        """Delete a paper"""
        if paper_id >= len(self.papers_df):
            return False
        
        self.papers_df = self.papers_df.drop(paper_id).reset_index(drop=True)
        self.save_papers()
        return True
    
    def get_all_topics(self) -> List[str]:
        """Get all unique topics"""
        all_topics = set()
        for topics_list in self.papers_df['Topics']:
            if isinstance(topics_list, list):
                all_topics.update(topics_list)
        return sorted(list(all_topics))

# Global database instance
db = PaperDatabase()
