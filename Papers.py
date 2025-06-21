import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime, timedelta
import random
import arxiv
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from collections import Counter

PAPER_FILE = "paper_with_topics.json"
READING_QUOTES = [
    # Classic Reading Quotes
    "📚 The more that you read, the more things you will know. The more that you learn, the more places you'll go. - Dr. Seuss",
    "📖 Today a reader, tomorrow a leader. - Margaret Fuller",
    "🤓 Reading is to the mind what exercise is to the body. - Joseph Addison",
    "🌟 Books are a uniquely portable magic. - Stephen King",
    "💡 Knowledge is power. - Francis Bacon",

    # Research & Scientific Quotes
    "🔬 Research is formalized curiosity. It is poking and prying with a purpose. - Zora Neale Hurston",
    "🧪 The important thing is not to stop questioning. Curiosity has its own reason for existence. - Albert Einstein",
    "🎯 Research is seeing what everybody else has seen and thinking what nobody else has thought. - Albert Szent-Györgyi",
    "📊 The good thing about science is that it's true whether or not you believe in it. - Neil deGrasse Tyson",
    "🔍 Research is creating new knowledge. - Neil Armstrong",

    # Learning & Growth Quotes
    "📘 Learning never exhausts the mind. - Leonardo da Vinci",
    "🌱 The capacity to learn is a gift; the ability to learn is a skill; the willingness to learn is a choice. - Brian Herbert",
    "💭 The beautiful thing about learning is that nobody can take it away from you. - B.B. King",
    "🎓 Education is not preparation for life; education is life itself. - John Dewey",
    "🌍 The world is a book and those who do not travel read only one page. - Saint Augustine",

    # Academic & Scholarly Quotes
    "📝 The art of writing is the art of discovering what you believe. - Gustave Flaubert",
    "📚 Study hard what interests you the most in the most undisciplined, irreverent and original manner possible. - Richard Feynman",
    "🎨 The noblest pleasure is the joy of understanding. - Leonardo da Vinci",
    "🔮 The future belongs to those who learn more skills and combine them in creative ways. - Robert Greene",
    "📖 Reading is an exercise in empathy; an exercise in walking in someone else's shoes for a while. - Malorie Blackman",

    # Motivation & Inspiration
    "⭐ The more I read, the more I acquire, the more certain I am that I know nothing. - Voltaire",
    "🌟 There is no friend as loyal as a book. - Ernest Hemingway",
    "💫 Once you learn to read, you will be forever free. - Frederick Douglass",
    "🚀 Reading is an act of civilization; it's one of the greatest acts of civilization because it takes the free raw material of the mind and builds castles of possibilities. - Ben Okri",
    "🎯 Reading is essential for those who seek to rise above the ordinary. - Jim Rohn",

    # Modern Perspectives
    "💻 The internet is 99% reading. - Kevin Kelly",
    "📱 In the age of information overload, the ability to read deeply and thoughtfully is a superpower. - Cal Newport",
    "🌐 Reading is still the main way that I both learn new things and test my understanding. - Bill Gates",
    "🔄 The more you read, the better you're going to become as a writer. - Junot Díaz",
    "📖 A reader lives a thousand lives before he dies. The man who never reads lives only one. - George R.R. Martin",

    # Wisdom & Understanding
    "🧠 Reading furnishes the mind only with materials of knowledge; it is thinking that makes what we read ours. - John Locke",
    "🎭 Reading is a conversation. All books talk. But a good book listens as well. - Mark Haddon",
    "🌅 The reading of all good books is like conversation with the finest minds of past centuries. - René Descartes",
    "🔮 Reading is a basic tool in the living of a good life. - Mortimer J. Adler",
    "🌱 A book is a garden, an orchard, a storehouse, a party, a company by the way, a counselor, a multitude of counselors. - Charles Baudelaire",

    # Personal Growth
    "🌈 Reading is a discount ticket to everywhere. - Mary Schmich",
    "🎨 Reading should not be presented to children as a chore, a duty. It should be offered as a gift. - Kate DiCamillo",
    "🌟 The person who deserves most pity is a lonesome one on a rainy day who doesn't know how to read. - Benjamin Franklin",
    "🚀 Reading is to the mind what exercise is to the body and prayer is to the soul. - Matthew Kelly",
    "💫 There are worse crimes than burning books. One of them is not reading them. - Joseph Brodsky"
]


def fetch_paper_details(arxiv_url):
    """Fetch paper details directly using requests with explicit SSL verification disabled."""
    try:
        import re
        import requests
        import xml.etree.ElementTree as ET
        from datetime import datetime
        
        # Suppress SSL warning messages
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Extract arXiv ID from URL
        arxiv_id = extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            return None, "Invalid arXiv URL format"
        
        # Make direct request to arXiv API with SSL verification disabled
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        
        try:
            response = requests.get(api_url, verify=False, timeout=10)
            
            if response.status_code != 200:
                return None, f"API returned status code {response.status_code}"
            
            # XML namespaces for arXiv API
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Check if paper exists
            entry = root.find('.//atom:entry', ns)
            if entry is None:
                return None, f"No paper found with ID {arxiv_id}"
            
            # Extract paper details
            title = entry.find('./atom:title', ns).text.strip()
            summary = entry.find('./atom:summary', ns).text.strip().replace('\n', ' ')
            
            # Get publication date
            published = entry.find('./atom:published', ns).text
            published_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
            
            # Extract categories/topics
            categories = []
            for category in entry.findall('./atom:category', ns):
                term = category.get('term')
                if term and '.' in term:
                    # Extract primary category (e.g., 'cs' from 'cs.CL')
                    primary = term.split('.')[0]
                    if primary not in categories:
                        categories.append(primary)
            
            # Create paper details dictionary
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


def extract_arxiv_id(url):
    """Extract arXiv ID from URL or return the ID if directly provided."""
    import re
    
    # Match patterns like arxiv.org/abs/2312.12345 or arxiv.org/pdf/2312.12345.pdf
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)',
        r'(\d{4}\.\d{5,6})'  # Direct arXiv ID format (handle both 4 and 5+ digit formats)
    ]

    for pattern in patterns:
        if match := re.search(pattern, url):
            return match.group(1)
    return None


def create_topic_evolution(paper_df):
    """Create a visualization showing how topics evolve over time."""
    if paper_df.empty or len(paper_df) < 3:
        return None

    # Ensure date is in datetime format
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')

    # Get all unique topics
    all_topics = set()
    for topics in paper_df['Topics']:
        if isinstance(topics, list):
            all_topics.update(topics)

    # Get top 5 topics
    topic_counts = Counter()
    for topics in paper_df['Topics']:
        if isinstance(topics, list):
            for topic in topics:
                topic_counts[topic] += 1

    top_topics = [topic for topic, count in topic_counts.most_common(5)]

    # Create time series data for each topic
    topic_time_series = {}

    # Group papers by month
    paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')
    monthly_papers = paper_df.groupby('Month')

    # For each month, count papers with each topic
    for month, group in monthly_papers:
        for topic in top_topics:
            if topic not in topic_time_series:
                topic_time_series[topic] = []

            # Count papers with this topic in this month
            count = sum(1 for topics in group['Topics'] if isinstance(topics, list) and topic in topics)
            topic_time_series[topic].append((month, count))

    # Convert to DataFrame suitable for plotting
    plot_data = []
    for topic, time_series in topic_time_series.items():
        for month, count in time_series:
            plot_data.append({
                'Month': month,
                'Topic': topic,
                'Count': count
            })

    if not plot_data:
        return None

    df = pd.DataFrame(plot_data)
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.sort_values('Month')

    fig = px.line(df, x='Month', y='Count', color='Topic',
                  title="Topic Evolution Over Time",
                  labels={'Count': 'Papers Count', 'Month': 'Time Period'})

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Number of Papers",
        legend_title="Research Topics",
        height=400
    )

    return fig


def calculate_research_impact_metrics(paper_df):
    """Calculate metrics that indicate research impact and efficiency."""
    if paper_df.empty:
        return {
            "knowledge_breadth": 0,
            "knowledge_depth": 0,
            "research_efficiency": 0,
            "topic_concentration": 0,
            "reading_consistency": 0,
            "exploration_vs_exploitation": 0,
            "research_velocity_trend": "stable"
        }

    # Count unique topics and their frequency
    topic_counts = Counter()
    for topics in paper_df['Topics']:
        if isinstance(topics, list):
            for topic in topics:
                topic_counts[topic] += 1

    total_papers = len(paper_df)
    unique_topics = len(topic_counts)

    # Calculate breadth (number of unique topics per paper)
    knowledge_breadth = unique_topics / max(1, total_papers)

    # Calculate depth (papers per topic)
    knowledge_depth = sum(topic_counts.values()) / max(1, unique_topics)

    # Calculate reading efficiency
    read_papers = len(paper_df[paper_df['Reading Status'] == 'Read'])
    reading_efficiency = read_papers / max(1, total_papers)

    # Calculate topic concentration (how concentrated on top topics)
    if topic_counts:
        top_topic_count = topic_counts.most_common(1)[0][1]
        topic_concentration = top_topic_count / max(1, sum(topic_counts.values()))
    else:
        topic_concentration = 0

    # Calculate reading consistency
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
    paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')
    monthly_counts = paper_df.groupby('Month').size()

    if len(monthly_counts) > 1:
        reading_consistency = 1 - (monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0)
        reading_consistency = max(0, min(1, reading_consistency))  # Ensure between 0 and 1
    else:
        reading_consistency = 0

    # Calculate exploration vs exploitation
    # (ratio of new topics to papers in existing topics)
    if len(paper_df) > 5:
        recent_papers = paper_df.sort_values('Date Added').tail(5)
        old_papers = paper_df.sort_values('Date Added').head(len(paper_df) - 5)

        old_topics = set()
        for topics in old_papers['Topics']:
            if isinstance(topics, list):
                old_topics.update(topics)

        new_topic_count = 0
        existing_topic_count = 0

        for topics in recent_papers['Topics']:
            if isinstance(topics, list):
                for topic in topics:
                    if topic in old_topics:
                        existing_topic_count += 1
                    else:
                        new_topic_count += 1

        exploration_ratio = new_topic_count / max(1, new_topic_count + existing_topic_count)
    else:
        exploration_ratio = 0.5  # Default to neutral if not enough data

    # Determine research velocity trend
    if len(monthly_counts) >= 3:
        recent_months = sorted(monthly_counts.index)[-3:]
        if len(recent_months) == 3:
            start_count = monthly_counts[recent_months[0]]
            end_count = monthly_counts[recent_months[2]]

            if end_count > start_count * 1.2:
                velocity_trend = "accelerating"
            elif end_count < start_count * 0.8:
                velocity_trend = "decelerating"
            else:
                velocity_trend = "stable"
        else:
            velocity_trend = "stable"
    else:
        velocity_trend = "stable"

    return {
        "knowledge_breadth": round(knowledge_breadth, 2),
        "knowledge_depth": round(knowledge_depth, 2),
        "research_efficiency": round(reading_efficiency, 2),
        "topic_concentration": round(topic_concentration, 2),
        "reading_consistency": round(reading_consistency, 2),
        "exploration_vs_exploitation": round(exploration_ratio, 2),
        "research_velocity_trend": velocity_trend
    }


def create_radar_chart(metrics):
    """Create a radar chart of research metrics."""
    categories = ['Knowledge Breadth', 'Knowledge Depth', 'Research Efficiency',
                  'Reading Consistency', 'Topic Focus', 'Exploration']

    values = [
        metrics["knowledge_breadth"] * 5,  # Scale to 0-5
        metrics["knowledge_depth"] / 2,  # Scale down if very high
        metrics["research_efficiency"] * 5,
        metrics["reading_consistency"] * 5,
        metrics["topic_concentration"] * 5,
        metrics["exploration_vs_exploitation"] * 5
    ]

    # Close the loop for the radar chart
    categories = categories + [categories[0]]
    values = values + [values[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Research Profile',
        line=dict(color='#1f77b4'),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        title="Research Profile Radar",
        height=400
    )

    return fig


def create_reading_forecast(paper_df):
    """Create a forecast of reading progress based on historical data."""
    if paper_df.empty or len(paper_df) < 3:
        return None, None

    # Ensure date is in datetime format
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
    paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')

    # Get monthly counts of papers added
    monthly_added = paper_df.groupby('Month').size()

    # Get monthly counts of papers read
    monthly_read = paper_df[paper_df['Reading Status'] == 'Read'].groupby('Month').size()

    # Merge the data
    monthly_data = pd.DataFrame({
        'Added': monthly_added,
        'Read': monthly_read
    }).fillna(0)

    # Calculate the average read rate
    read_rate = monthly_data['Read'].mean() / max(1, monthly_data['Added'].mean())
    read_rate = min(1.0, read_rate)  # Cap at 1.0 (can't read more than 100%)

    # Calculate average papers added per month
    avg_added = monthly_data['Added'].mean()

    # Calculate the current backlog
    total_papers = len(paper_df)
    read_papers = len(paper_df[paper_df['Reading Status'] == 'Read'])
    backlog = total_papers - read_papers

    # Calculate months to clear backlog
    if read_rate * avg_added > 0:
        months_to_clear = backlog / (read_rate * avg_added)
    else:
        months_to_clear = float('inf')

    # Simple forecast model
    current_date = datetime.now()
    forecast_months = 6

    forecast_dates = [current_date + timedelta(days=30 * i) for i in range(forecast_months)]
    forecast_dates_str = [d.strftime('%Y-%m') for d in forecast_dates]

    # Project based on historical trends
    forecast_added = [round(avg_added) for _ in range(forecast_months)]
    forecast_backlog = [backlog]

    for i in range(forecast_months - 1):
        new_backlog = forecast_backlog[-1] + forecast_added[i] - (avg_added * read_rate)
        new_backlog = max(0, new_backlog)  # Can't have negative backlog
        forecast_backlog.append(new_backlog)

    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'Month': forecast_dates_str,
        'Projected Added': forecast_added,
        'Projected Backlog': forecast_backlog
    })

    # Create a visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=forecast_df['Month'],
        y=forecast_df['Projected Added'],
        name='Projected New Papers',
        marker_color='lightskyblue'
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df['Month'],
        y=forecast_df['Projected Backlog'],
        name='Projected Backlog',
        marker_color='red',
        mode='lines+markers'
    ))

    fig.update_layout(
        title="6-Month Research Forecast",
        xaxis_title="Month",
        yaxis_title="Paper Count",
        legend_title="Metrics",
        height=400
    )

    forecast_metrics = {
        "current_backlog": int(backlog),
        "read_rate": round(read_rate * 100, 1),
        "avg_monthly_papers": round(avg_added, 1),
        "months_to_clear": round(months_to_clear, 1) if months_to_clear != float('inf') else "∞",
        "backlog_trend": "increasing" if forecast_backlog[-1] > backlog else "decreasing" if forecast_backlog[
                                                                                                 -1] < backlog else "stable"
    }

    return fig, forecast_metrics


def generate_research_insights(paper_df, analysis_data, impact_metrics, forecast_metrics=None):
    """Generate data-driven research insights."""
    insights = []

    if paper_df.empty or len(paper_df) < 3:
        return ["Add at least 3 papers to generate meaningful insights."]

    # Reading progress insights
    completion_rate = analysis_data["completion_rate"]
    if completion_rate < 30:
        insights.append({
            "category": "Reading Progress",
            "insight": f"Low completion rate ({completion_rate:.1f}%)",
            "recommendation": "Focus on completing existing papers before adding new ones",
            "priority": "high" if completion_rate < 20 else "medium"
        })
    elif completion_rate > 70:
        insights.append({
            "category": "Reading Progress",
            "insight": f"High completion rate ({completion_rate:.1f}%)",
            "recommendation": "You're effectively completing papers. Consider increasing your research scope",
            "priority": "low"
        })

    # Topic diversity insights
    if impact_metrics["knowledge_breadth"] < 0.3 and len(paper_df) > 5:
        insights.append({
            "category": "Research Focus",
            "insight": "Low topic diversity relative to paper count",
            "recommendation": "Consider exploring adjacent research areas to broaden your knowledge",
            "priority": "medium"
        })
    elif impact_metrics["knowledge_breadth"] > 1.5:
        insights.append({
            "category": "Research Focus",
            "insight": "Very high topic diversity",
            "recommendation": "Consider focusing on fewer areas for deeper expertise",
            "priority": "medium"
        })

    # Topic concentration insights
    if impact_metrics["topic_concentration"] > 0.5:
        top_topic = max(analysis_data["topic_distribution"].items(), key=lambda x: x[1])[0] if analysis_data[
            "topic_distribution"] else "None"
        insights.append({
            "category": "Research Specialization",
            "insight": f"Strong specialization in '{top_topic}'",
            "recommendation": "Your focused approach may lead to expertise. Consider exploring how this topic connects to others",
            "priority": "low"
        })

    # Reading consistency insights
    if impact_metrics["reading_consistency"] < 0.3 and len(paper_df) > 5:
        insights.append({
            "category": "Reading Habits",
            "insight": "Inconsistent reading pattern with large variations between periods",
            "recommendation": "Develop a more consistent research schedule for better knowledge retention",
            "priority": "medium"
        })

    # Research velocity insights
    if impact_metrics["research_velocity_trend"] == "decelerating" and len(paper_df) > 5:
        insights.append({
            "category": "Research Momentum",
            "insight": "Your research pace is slowing down",
            "recommendation": "Set specific goals for paper discovery and reading to maintain momentum",
            "priority": "medium"
        })
    elif impact_metrics["research_velocity_trend"] == "accelerating":
        insights.append({
            "category": "Research Momentum",
            "insight": "Accelerating research velocity",
            "recommendation": "Ensure quality isn't sacrificed for quantity by maintaining thorough notes",
            "priority": "low"
        })


def load_paper():
    """
    Load and validate the papers JSON file, with error handling and recovery.
    Returns a DataFrame with proper column structure.
    """
    try:
        # First try normal loading
        if os.path.exists(PAPER_FILE):
            with open(PAPER_FILE, 'r') as f:
                try:
                    paper_dict = json.load(f)
                except json.JSONDecodeError:
                    # If JSON is corrupted, try to recover the file
                    print(f"Warning: {PAPER_FILE} is corrupted. Creating new file.")
                    paper_dict = []

                    # Backup the corrupted file
                    if os.path.exists(PAPER_FILE):
                        backup_name = f"{PAPER_FILE}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        try:
                            os.rename(PAPER_FILE, backup_name)
                            print(f"Corrupted file backed up as {backup_name}")
                        except Exception as e:
                            print(f"Failed to create backup: {str(e)}")
        else:
            paper_dict = []

        # Convert to DataFrame and ensure proper structure
        paper_df = pd.DataFrame(paper_dict)

        # Initialize DataFrame with required columns if empty
        if paper_df.empty:
            paper_df = pd.DataFrame(columns=[
                'Title', 'Reading Status', 'Date Added', 'Link', 'Topics', 'Description'
            ])

        # Ensure all required columns exist
        required_columns = {
            'Title': str,
            'Reading Status': str,
            'Date Added': str,
            'Link': str,
            'Topics': list,
            'Description': str
        }

        for col, dtype in required_columns.items():
            if col not in paper_df.columns:
                paper_df[col] = dtype() if dtype != list else [[] for _ in range(len(paper_df))]

        # Clean up Topics column
        paper_df['Topics'] = paper_df['Topics'].apply(
            lambda x: [t.strip() for t in (x if isinstance(x, list) else
                                           (x.split(',') if isinstance(x, str) and x else []))]
        )

        # Ensure Date Added is in correct format
        paper_df['Date Added'] = paper_df['Date Added'].apply(
            lambda x: datetime.today().strftime('%Y-%m-%d') if pd.isna(x) else x
        )

        # Save the cleaned DataFrame back to file
        save_paper(paper_df)

        return paper_df

    except Exception as e:
        print(f"Error loading paper file: {str(e)}")
        # Return empty DataFrame with proper structure
        return pd.DataFrame(columns=[
            'Title', 'Reading Status', 'Date Added', 'Link', 'Topics', 'Description'
        ])


def serialize_dates(obj):
    """Custom JSON serializer for handling datetime and Timestamp objects."""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.strftime('%Y-%m-%d')
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def safe_date_parse(date_str):
    """Safely parse date strings, returning None for NaT values."""
    try:
        date = pd.to_datetime(date_str)
        return date.date() if pd.notnull(date) else datetime.today().date()
    except Exception as e:
        return datetime.today().date()


def save_paper(paper_df):
    if 'Date Added' in paper_df.columns:
        # Convert 'Date Added' column to string format before saving
        paper_df = paper_df.copy()
        paper_df['Date Added'] = paper_df['Date Added'].apply(
            lambda x: x.strftime('%Y-%m-%d') if isinstance(x, (datetime, pd.Timestamp)) else str(x)
        )
    paper_dict = paper_df.to_dict('records')
    with open(PAPER_FILE, 'w') as f:
        json.dump(paper_dict, f, default=serialize_dates)


def reset_form():
    st.session_state.form_reset = True
    st.session_state['title'] = ""
    st.session_state['topics'] = ""
    st.session_state['link'] = "https://arxiv.org/abs/"
    st.session_state['description'] = ""
    st.session_state['edit_title'] = ""
    st.session_state['edit_link'] = ""
    st.session_state['edit_topics'] = ""
    st.session_state['edit_description'] = ""


def add_paper(title, reading_status, date_added, link, topics, description):
    # Check if paper with same title exists (case-insensitive)
    existing_titles = st.session_state.paper['Title'].str.lower()
    if title.lower() in existing_titles.values:
        st.error(f"🚫 A paper with title '{title}' already exists!")
        return False

    # If title doesn't exist, proceed with adding the paper
    topics_list = topics if isinstance(topics, list) else []
    new_paper = pd.DataFrame({
        'Title': [title],
        'Reading Status': [reading_status],
        'Date Added': [date_added],
        'Link': [link],
        'Topics': [topics_list],
        'Description': [description]
    })
    st.session_state.paper = pd.concat([st.session_state.paper, new_paper], ignore_index=True)
    st.session_state.paper = st.session_state.paper.sort_values('Title')
    save_paper(st.session_state.paper)
    return True


def edit_paper(index, new_title, new_status, new_date, new_link, new_topics, new_description):
    # Convert topics string to list
    topics_list = [t.strip() for t in new_topics.split(",") if t.strip()]
    st.session_state.paper.at[index, 'Title'] = new_title
    st.session_state.paper.at[index, 'Reading Status'] = new_status
    st.session_state.paper.at[index, 'Date Added'] = new_date
    st.session_state.paper.at[index, 'Link'] = new_link
    st.session_state.paper.at[index, 'Topics'] = topics_list
    st.session_state.paper.at[index, 'Description'] = new_description
    save_paper(st.session_state.paper)
    reset_form()


def display_topics(topics):
    if isinstance(topics, list):
        return ", ".join(topics)
    return str(topics)


def get_reading_streak(paper_df):
    if 'Date Added' not in paper_df.columns or paper_df.empty:
        return 0

    try:
        paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], format='%Y-%m-%d', errors='coerce')
    except ValueError:
        # If some values fail to parse, `errors='coerce'` will set them to NaT
        st.warning("Some dates could not be parsed and were set to NaT.")

    last_30_days = datetime.now() - pd.Timedelta(days=30)
    recent_papers = paper_df[paper_df['Date Added'] >= last_30_days]
    return len(recent_papers)


def create_progress_bar(current, total, title):
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{title}: {current}/{total}")


def analyze_reading_habits(paper_df):
    """Generate analysis metrics for reading habits."""
    if paper_df.empty:
        return {
            "total_papers": 0,
            "read_papers": 0,
            "reading_papers": 0,
            "want_to_read_papers": 0,
            "reading_velocity": 0,
            "avg_reading_time": 0,
            "topic_distribution": {},
            "monthly_activity": {},
            "completion_rate": 0,
            "recent_topics": []
        }

    # Ensure date format is consistent
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')

    # Basic counts
    total_papers = len(paper_df)
    read_papers = len(paper_df[paper_df['Reading Status'] == 'Read'])
    reading_papers = len(paper_df[paper_df['Reading Status'] == 'Reading'])
    want_to_read_papers = len(paper_df[paper_df['Reading Status'] == 'Want to Read'])

    # Reading velocity (papers per month)
    if total_papers > 0:
        earliest_date = paper_df['Date Added'].min()
        latest_date = paper_df['Date Added'].max()

        # Calculate months between earliest and latest dates
        if pd.notnull(earliest_date) and pd.notnull(latest_date):
            months_diff = (latest_date.year - earliest_date.year) * 12 + (latest_date.month - earliest_date.month)
            months_diff = max(1, months_diff)  # Avoid division by zero
            reading_velocity = total_papers / months_diff
        else:
            reading_velocity = 0
    else:
        reading_velocity = 0

    # Average estimated reading time (assuming 1 paper takes 2 days to read)
    avg_reading_time = 2  # days per paper

    # Topic distribution
    topic_counts = Counter()
    for topics in paper_df['Topics']:
        if isinstance(topics, list):
            for topic in topics:
                topic_counts[topic] += 1

    # Sort topics by frequency
    topic_distribution = dict(topic_counts.most_common())

    # Monthly activity
    paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')
    monthly_activity = paper_df.groupby('Month').size().to_dict()

    # Completion rate
    completion_rate = (read_papers / total_papers * 100) if total_papers > 0 else 0

    # Recent trends in topics (last 3 months)
    three_months_ago = datetime.now() - timedelta(days=90)
    recent_papers = paper_df[paper_df['Date Added'] >= three_months_ago]
    recent_topic_counts = Counter()

    for topics in recent_papers['Topics']:
        if isinstance(topics, list):
            for topic in topics:
                recent_topic_counts[topic] += 1

    recent_topics = [topic for topic, _ in recent_topic_counts.most_common(5)]

    return {
        "total_papers": total_papers,
        "read_papers": read_papers,
        "reading_papers": reading_papers,
        "want_to_read_papers": want_to_read_papers,
        "reading_velocity": round(reading_velocity, 2),
        "avg_reading_time": avg_reading_time,
        "topic_distribution": topic_distribution,
        "monthly_activity": monthly_activity,
        "completion_rate": round(completion_rate, 1),
        "recent_topics": recent_topics
    }


def create_reading_timeline(paper_df):
    """Create a timeline visualization of papers added over time."""
    if paper_df.empty:
        return None

    # Ensure date is in datetime format
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')

    # Create a copy with only valid dates
    df_valid = paper_df.dropna(subset=['Date Added'])

    if df_valid.empty:
        return None

    # Sort by date
    df_valid = df_valid.sort_values('Date Added')

    # Create timeline
    fig = px.scatter(
        df_valid,
        x='Date Added',
        y=[1] * len(df_valid),  # All papers at same level
        color='Reading Status',
        hover_name='Title',
        size=[10] * len(df_valid),  # Consistent dot size
        color_discrete_map={
            'Read': '#4CAF50',  # Green
            'Reading': '#FFC107',  # Amber
            'Want to Read': '#2196F3'  # Blue
        },
        title="Paper Reading Timeline",
        labels={'Reading Status': 'Status'}
    )

    # Improve layout
    fig.update_layout(
        yaxis_title="",
        yaxis_showticklabels=False,
        yaxis_showgrid=False,
        height=300,
        hovermode="closest"
    )

    return fig


def create_topic_network(paper_df):
    """Create a network visualization of related topics."""
    if paper_df.empty:
        return None

    # Extract all unique topics
    all_topics = set()
    for topics in paper_df['Topics']:
        if isinstance(topics, list):
            all_topics.update(topics)

    if not all_topics:
        return None

    # Count co-occurrences of topics
    topic_pairs = []
    for topics in paper_df['Topics']:
        if isinstance(topics, list) and len(topics) > 1:
            for i, topic1 in enumerate(topics):
                for topic2 in topics[i + 1:]:
                    topic_pairs.append((topic1, topic2))

    # Count frequency of pairs
    pair_counts = Counter(topic_pairs)

    # Create nodes for each topic
    nodes = list(all_topics)
    node_sizes = []

    for topic in nodes:
        # Count papers with this topic
        count = sum(1 for topics in paper_df['Topics'] if isinstance(topics, list) and topic in topics)
        node_sizes.append(count * 10)  # Scale node size

    # Create edges between topics that co-occur
    edge_x = []
    edge_y = []
    edge_weights = []

    # Simple circular layout for nodes
    node_positions = {}
    n = len(nodes)
    for i, node in enumerate(nodes):
        angle = 2 * np.pi * i / n
        node_positions[node] = (np.cos(angle), np.sin(angle))

    for (topic1, topic2), count in pair_counts.items():
        if count > 0:  # Only include edges with connections
            x0, y0 = node_positions[topic1]
            x1, y1 = node_positions[topic2]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(count)

    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Create node trace
    node_x = [pos[0] for pos in node_positions.values()]
    node_y = [pos[1] for pos in node_positions.values()]
    
    # Calculate node colors based on paper count
    node_adjacencies = []
    for node in nodes:
        count = sum(1 for topics in paper_df['Topics'] if isinstance(topics, list) and node in topics)
        node_adjacencies.append(count)

    # Create node trace with a simplified marker configuration
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=nodes,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            color=node_adjacencies,  # Set color at initialization
            line_width=2))
    
    # Skip the colorbar for now to simplify

    # Create figure with corrected title format
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text='Topic Relationship Network',
                            font=dict(size=16)
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    ))

    return fig

def main():
    # Set page config with a fun emoji and custom theme
    st.set_page_config(
        page_title="📚 Research Reading Tracker",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .element-container {
            margin-bottom: 1rem;
        }
        .stProgress .st-bo {
            background-color: #f0f2f6;
        }
        .stProgress .st-bp {
            background: linear-gradient(to right, #ff4b4b, #ff9f1c);
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar with statistics and motivation
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=📚", width=150)
        st.title("Your Reading Journey")

        if 'paper' in st.session_state:
            total_papers = len(st.session_state.paper)
            read_papers = len(st.session_state.paper[st.session_state.paper['Reading Status'] == 'Read'])
            reading_papers = len(st.session_state.paper[st.session_state.paper['Reading Status'] == 'Reading'])
            reading_streak = get_reading_streak(st.session_state.paper)

            st.markdown("### 📊 Your Stats")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Papers", total_papers)
                st.metric("Reading Streak", f"{reading_streak} days")
            with col2:
                st.metric("Completed", read_papers)
                st.metric("In Progress", reading_papers)

        st.markdown("---")
        st.markdown(f"### ✨ Daily Inspiration")
        st.markdown(f"*{random.choice(READING_QUOTES)}*")

    if 'paper' not in st.session_state:
        st.session_state.paper = load_paper()

        # Initialize success message state if not exists
    if 'success_message' not in st.session_state:
        st.session_state.success_message = None

    tab1, tab2, tab3 = st.tabs(["➕ Add Paper", "📋 View Collection", "📊 Analysis"])

    with tab1:
        # Display success message if exists
        if st.session_state.success_message:
            st.success(st.session_state.success_message)
            # Clear the message after displaying
            st.session_state.success_message = None

        # Create a separate form for the fetch functionality
        with st.form(key="fetch_form"):
            url_col1, url_col2 = st.columns([3, 1])
            with url_col1:
                arxiv_url = st.text_input(
                    "ArXiv URL",
                    value="https://arxiv.org/abs/"
                )

            with url_col2:
                fetch_submitted = st.form_submit_button("🔍 Fetch Details")

            if fetch_submitted:
                if arxiv_url:
                    paper_details, error = fetch_paper_details(arxiv_url)
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.session_state['paper_details'] = paper_details
                        st.session_state['title'] = paper_details['title']
                        st.session_state['link'] = paper_details['link']
                        st.session_state['description'] = paper_details['description']
                        st.session_state.success_message = "📄 Paper details fetched successfully!"
                        st.rerun()
                else:
                    st.warning("Please enter an arXiv URL")

        # Get existing topics for suggestions
        all_topics = set(topic for topics_list in st.session_state.paper['Topics'] for topic in topics_list if
                        isinstance(topics_list, list))
        sorted_topics = sorted(list(all_topics))

        # Separate form for adding paper details
        with st.form(key="add_paper_form"):
            title = st.text_input("📕 Paper Title",
                                value=st.session_state.get('paper_details', {}).get('title', ''),
                                placeholder="Enter the paper title...")

            col1, col2 = st.columns(2)
            with col1:
                date_added = st.date_input(
                    "📅 Date Added",
                    value=datetime.strptime(
                        st.session_state.get('paper_details', {}).get('date', datetime.today().strftime('%Y-%m-%d')),
                        '%Y-%m-%d'
                    ).date()
                )
                reading_status = st.selectbox(
                    "📖 Reading Status",
                    ["Want to Read", "Reading", "Read"]
                )

            with col2:
                link = st.text_input("🌐 Link",
                                    value=st.session_state.get('paper_details', {}).get('link', ''),
                                    help="ArXiv or DOI link")
                
                # Multi-select for topics with autocomplete
                if sorted_topics:
                    # Initialize default topics from session state if available
                    default_topics = []
                    if st.session_state.get('paper_details', {}).get('topics'):
                        paper_topics = st.session_state.get('paper_details', {}).get('topics')
                        # Filter to only include topics that exist in sorted_topics
                        default_topics = [topic for topic in paper_topics if topic in sorted_topics]
                    
                    selected_topics = st.multiselect(
                        "🏷️ Topics",
                        options=sorted_topics,
                        default=default_topics,
                        help="Select existing topics or type to create new ones"
                    )
                    
                    # Allow custom topics with a text input field
                    custom_topics = st.text_input(
                        "Add Custom Topics",
                        placeholder="Type new topics (comma-separated)",
                        help="Add topics not in the suggestion list"
                    )
                    
                    # Combine selected and custom topics
                    topic_list = selected_topics.copy()
                    if custom_topics:
                        custom_topic_list = [t.strip() for t in custom_topics.split(",") if t.strip()]
                        topic_list.extend(custom_topic_list)
                else:
                    # Fallback to regular text input if no existing topics
                    topics = st.text_input("🏷️ Topics",
                                        placeholder="AI, ML, NLP...",
                                        help="Comma-separated topics")
                    topic_list = [t.strip() for t in topics.split(",") if t.strip()]

            description = st.text_area(
                "📝 Notes & Thoughts",
                value=st.session_state.get('paper_details', {}).get('description', ''),
                placeholder="What makes this paper interesting? Key takeaways? Future ideas?"
            )

            # Add form submit button
            submitted = st.form_submit_button("📚 Add to Collection")
            if submitted:
                if not title:
                    st.error("Please enter a paper title!")
                else:
                    if add_paper(title, reading_status,
                                date_added.strftime('%Y-%m-%d'),
                                link, topic_list, description):
                        st.session_state.success_message = "🎉 Paper added successfully!"
                        # Clear the paper details after successful addition
                        st.session_state.paper_details = {}
                        st.session_state.title = ""
                        st.session_state.link = ""
                        st.session_state.description = ""
                        st.rerun()

                        
    with tab2:
        st.markdown("### 📚 Your Paper Collection")

        # Enhanced filters section
        with st.expander("🔍 Advanced Filters", expanded=True):
            filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])

            with filter_col1:
                all_topics = set(topic for topics_list in st.session_state.paper['Topics'] for topic in topics_list if
                                 isinstance(topics_list, list))
                topic_filter = st.multiselect("🏷️ Filter by Topics", options=["All"] + sorted(list(all_topics)))

            with filter_col2:
                status_options = ["Want to Read", "Reading", "Read"]
                status_filter = st.radio("📖 Reading Status", options=["All"] + status_options, horizontal=True)

            with filter_col3:
                text_filter = st.text_input("🔍 Search by Title/Description/Topic",
                                            placeholder="Enter keywords to search...")

        date_range = None

        # Sorting options
        sort_col1, sort_col2 = st.columns([1, 1])
        with sort_col1:
            sort_by = st.selectbox("🔄 Sort by", ["Date Added", "Title", "Reading Status", "Topics Count"])
        with sort_col2:
            sort_direction = st.radio("⬆️⬇️ Order", ["Descending", "Ascending"], horizontal=True)

        # Filtering logic
        papers_to_display = st.session_state.paper.copy()

        # Apply status filter
        if status_filter != "All":
            papers_to_display = papers_to_display[papers_to_display['Reading Status'] == status_filter]

        # Apply topic filter
        if topic_filter and "All" not in topic_filter:
            papers_to_display = papers_to_display[
                papers_to_display['Topics'].apply(
                    lambda x: any(topic in x for topic in topic_filter) if isinstance(x, list) else False)]

        # Apply date range filter
        if date_range and len(date_range) == 2:
            papers_to_display['Date Added'] = pd.to_datetime(papers_to_display['Date Added'])
            start_date, end_date = date_range
            papers_to_display = papers_to_display[(papers_to_display['Date Added'].dt.date >= start_date) &
                                                  (papers_to_display['Date Added'].dt.date <= end_date)]

        # Apply text filter
        if text_filter:
            papers_to_display = papers_to_display[
                papers_to_display['Title'].str.contains(text_filter, case=False, na=False) |
                papers_to_display['Description'].str.contains(text_filter, case=False, na=False) |
                papers_to_display['Topics'].apply(
                    lambda x: any(text_filter.lower() in t.lower() for t in x if isinstance(x, list)))
                ]

        # Calculate topics count for sorting
        papers_to_display['Topics Count'] = papers_to_display['Topics'].apply(
            lambda x: len(x) if isinstance(x, list) else 0)

        # Apply sorting
        ascending = sort_direction == "Ascending"
        if sort_by == "Date Added":
            papers_to_display['Date Added'] = pd.to_datetime(papers_to_display['Date Added'])
            papers_to_display = papers_to_display.sort_values('Date Added', ascending=ascending)
        elif sort_by == "Title":
            papers_to_display = papers_to_display.sort_values('Title', ascending=ascending)
        elif sort_by == "Reading Status":
            status_order = {"Read": 0, "Reading": 1, "Want to Read": 2}
            if ascending:
                status_order = {k: -v for k, v in status_order.items()}
            papers_to_display['Status Order'] = papers_to_display['Reading Status'].map(status_order)
            papers_to_display = papers_to_display.sort_values('Status Order', ascending=True)
        elif sort_by == "Topics Count":
            papers_to_display = papers_to_display.sort_values('Topics Count', ascending=ascending)

        # Paper edit form
        if hasattr(st.session_state, 'edit_mode') and st.session_state.edit_mode and hasattr(st.session_state,
                                                                                             'selected_paper_index'):
            st.markdown("### ✏️ Edit Paper")
            index = st.session_state.selected_paper_index
            paper = st.session_state.paper.loc[index]

            with st.form(key=f"edit_form_{index}"):
                col1, col2 = st.columns([1, 1])

                with col1:
                    new_title = st.text_input("Title", value=paper['Title'])
                    new_status = st.selectbox(
                        "Status",
                        ["Want to Read", "Reading", "Read"],
                        index=["Want to Read", "Reading", "Read"].index(paper['Reading Status'])
                    )
                    new_date = st.date_input(
                        "Date Added",
                        value=safe_date_parse(paper['Date Added'])
                    )

                with col2:
                    new_link = st.text_input("Link", value=paper['Link'])
                    new_topics = st.text_input("Topics", value=", ".join(paper['Topics']) if isinstance(paper['Topics'],
                                                                                                        list) else "")

                new_description = st.text_area("Notes", value=paper['Description'], height=150)

                col1, col2 = st.columns([1, 1])
                with col1:
                    submitted = st.form_submit_button("💾 Save Changes")
                with col2:
                    cancel = st.form_submit_button("❌ Cancel")

                if submitted:
                    # Update the paper
                    st.session_state.paper.at[index, 'Title'] = new_title
                    st.session_state.paper.at[index, 'Reading Status'] = new_status
                    st.session_state.paper.at[index, 'Date Added'] = new_date.strftime('%Y-%m-%d')
                    st.session_state.paper.at[index, 'Link'] = new_link
                    st.session_state.paper.at[index, 'Topics'] = [t.strip() for t in new_topics.split(",") if t.strip()]
                    st.session_state.paper.at[index, 'Description'] = new_description
                    save_paper(st.session_state.paper)
                    st.success("✨ Changes saved successfully!")
                    st.session_state.edit_mode = False
                    st.rerun()

                if cancel:
                    st.session_state.edit_mode = False
                    st.rerun()
        else:
            # Paper display section
            st.markdown(f"**Showing {len(papers_to_display)} papers**")

            if papers_to_display.empty:
                st.info("No papers match your filter criteria.")
            else:
                # Display papers in expandable card view with custom styling
                for idx, paper in papers_to_display.iterrows():
                    status_emoji = {
                        "Want to Read": "🔹",
                        "Reading": "🔶",
                        "Read": "✅"
                    }.get(paper['Reading Status'], "📄")

                    # Format the card header with title and date
                    paper_title = paper['Title'] if not pd.isna(paper['Title']) else "Untitled Paper"
                    date_str = pd.to_datetime(paper['Date Added']).strftime('%Y-%m-%d') if not pd.isna(
                        paper['Date Added']) else ""
                    card_header = f"{status_emoji} {paper_title} ({date_str})"

                    # Create the expandable card
                    with st.expander(card_header):
                        # Card content
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**Status:** {paper['Reading Status']}")

                            # Display topics if they exist
                            topics_str = ", ".join(paper['Topics']) if isinstance(paper['Topics'], list) and paper[
                                'Topics'] else "No topics"
                            st.markdown(f"**Topics:** {topics_str}")

                            # Display link if it exists
                            if paper['Link'] and not pd.isna(paper['Link']):
                                st.markdown(f"[Open Paper]({paper['Link']})")

                        with col2:
                            if st.button(f"✏️ Edit", key=f"edit_card_{idx}"):
                                st.session_state.selected_paper_index = idx
                                st.session_state.edit_mode = True
                                st.rerun()

                        # Notes section
                        st.markdown("#### 📝 Notes")
                        if paper['Description'] and not pd.isna(paper['Description']):
                            st.markdown(paper['Description'])
                        else:
                            st.info("No notes added yet.")

    with tab3:
        st.markdown("### 📊 Research Analysis Dashboard")

        # Get analysis data
        analysis_data = analyze_reading_habits(st.session_state.paper)

        # Display summary metrics in a more academic context
        st.subheader("📈 Research Metrics Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Research Corpus", analysis_data["total_papers"])
        with col2:
            st.metric("Completion Rate", f"{analysis_data['completion_rate']}%")
        with col3:
            st.metric("Research Velocity", f"{analysis_data['reading_velocity']} papers/month")
        with col4:
            if analysis_data["read_papers"] > 0:
                time_to_insights = f"{(analysis_data['avg_reading_time'] * analysis_data['total_papers']) / analysis_data['read_papers']:.1f} days"
            else:
                time_to_insights = "N/A"
            st.metric("Time to Insights", time_to_insights)

        # Research progress tracking
        st.subheader("📚 Research Progress")
        progress_cols = st.columns(3)
        with progress_cols[0]:
            st.markdown("##### Papers by Status")
            status_labels = ["Read", "Reading", "Want to Read"]
            status_values = [analysis_data["read_papers"], analysis_data["reading_papers"],
                             analysis_data["want_to_read_papers"]]
            status_fig = px.pie(names=status_labels, values=status_values,
                                color=status_labels,
                                color_discrete_map={'Read': '#4CAF50', 'Reading': '#FFC107', 'Want to Read': '#2196F3'},
                                hole=0.4)
            status_fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
            st.plotly_chart(status_fig, use_container_width=True)

        with progress_cols[1]:
            st.markdown("##### Research Exploration vs. Focus")
            if analysis_data["total_papers"] > 0:
                # Calculate impact metrics using the provided function
                impact_metrics = calculate_research_impact_metrics(st.session_state.paper)

                metrics_data = pd.DataFrame({
                    'Metric': ['Knowledge Breadth', 'Knowledge Depth', 'Research Efficiency', 'Topic Concentration',
                               'Reading Consistency', 'Exploration Ratio'],
                    'Value': [
                        f"{impact_metrics['knowledge_breadth']:.2f}",
                        f"{impact_metrics['knowledge_depth']:.2f}",
                        f"{impact_metrics['research_efficiency']:.2f}",
                        f"{impact_metrics['topic_concentration']:.2f}",
                        f"{impact_metrics['reading_consistency']:.2f}",
                        f"{impact_metrics['exploration_vs_exploitation']:.2f}"
                    ]
                })
                st.dataframe(metrics_data, hide_index=True, use_container_width=True)
            else:
                st.info("Add papers to see research focus metrics")

        with progress_cols[2]:
            st.markdown("##### Research Momentum")
            if analysis_data["monthly_activity"]:
                last_three_months = sorted(analysis_data["monthly_activity"].items())[-3:]
                if len(last_three_months) == 3:
                    momentum = ((last_three_months[2][1] - last_three_months[0][1]) /
                                max(1, last_three_months[0][1])) * 100
                    momentum_text = f"{momentum:.1f}% change over 3 months"
                else:
                    momentum_text = "Insufficient data"

                recent_activity = pd.DataFrame({
                    'Month': [m[0] for m in last_three_months],
                    'Papers': [m[1] for m in last_three_months]
                })
                st.dataframe(recent_activity, hide_index=True, use_container_width=True)

                # Use the velocity trend from impact metrics
                if 'impact_metrics' in locals():
                    st.markdown(
                        f"**Research Velocity Trend:** {impact_metrics['research_velocity_trend'].capitalize()}")
                else:
                    st.markdown(f"**Research Momentum:** {momentum_text}")
            else:
                st.info("Add more papers to track research momentum")

        # Research profile radar chart
        st.subheader("🔍 Research Profile")
        radar_cols = st.columns([2, 1])
        with radar_cols[0]:
            if analysis_data["total_papers"] > 0:
                # Use the provided function to create the radar chart
                radar_fig = create_radar_chart(impact_metrics)
                st.plotly_chart(radar_fig, use_container_width=True)
            else:
                st.info("Add papers to see your research profile")

        with radar_cols[1]:
            # Research domain statistics
            if analysis_data["topic_distribution"]:
                st.markdown("##### Top Research Areas")
                top_topics = dict(sorted(analysis_data["topic_distribution"].items(),
                                         key=lambda x: x[1], reverse=True)[:5])

                topic_strength = pd.DataFrame({
                    'Domain': list(top_topics.keys()),
                    'Papers': list(top_topics.values()),
                    'Expertise': [f"{(count / analysis_data['total_papers'] * 100):.1f}%"
                                  for count in top_topics.values()]
                })
                st.dataframe(topic_strength, hide_index=True, use_container_width=True)

                # Research diversity index calculation
                if len(analysis_data["topic_distribution"]) > 1:
                    total = sum(analysis_data["topic_distribution"].values())
                    diversity = -sum((count / total) * np.log(count / total)
                                     for count in analysis_data["topic_distribution"].values())
                    normalized_diversity = diversity / np.log(len(analysis_data["topic_distribution"]))
                    st.metric("Research Diversity Index", f"{normalized_diversity:.2f}")
                else:
                    st.metric("Research Diversity Index", "N/A")
            else:
                st.info("Add topics to your papers to analyze research domains")

        # Display citation network and paper relationships
        st.subheader("🔬 Research Knowledge Graph")
        knowledge_cols = st.columns([2, 1])
        with knowledge_cols[0]:
            # Topic network visualization with improved academic context
            network_fig = create_topic_network(st.session_state.paper)
            if network_fig:
                network_fig.update_layout(title="Research Domain Relationship Network")
                st.plotly_chart(network_fig, use_container_width=True)
            else:
                st.info("Add more papers with multiple topics to visualize your research domain network")

        with knowledge_cols[1]:
            # Topic evolution info
            st.markdown("##### Topic Evolution")
            if analysis_data["total_papers"] > 2:
                st.write("See the evolution of research topics over time in the chart below.")
            else:
                st.info("Add at least 3 papers to see topic evolution")

        # Topic evolution over time
        topic_evolution_fig = create_topic_evolution(st.session_state.paper)
        if topic_evolution_fig:
            st.plotly_chart(topic_evolution_fig, use_container_width=True)
        else:
            st.info("Add more papers with topics and dates to visualize topic evolution")

        # Research forecast section
        st.subheader("📅 Research Forecast & Planning")
        forecast_cols = st.columns([3, 1])
        with forecast_cols[0]:
            # Use the provided function to create the reading forecast
            forecast_fig, forecast_metrics = create_reading_forecast(st.session_state.paper)
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
            else:
                st.info("Add more papers with dates to generate research forecasts")

        with forecast_cols[1]:
            # Display forecast metrics
            if forecast_metrics:
                st.markdown("##### Research Forecast Metrics")
                forecast_data = pd.DataFrame({
                    'Metric': ['Current Backlog', 'Reading Rate', 'Avg Monthly Papers', 'Months to Clear Backlog',
                               'Backlog Trend'],
                    'Value': [
                        f"{forecast_metrics['current_backlog']}",
                        f"{forecast_metrics['read_rate']}%",
                        f"{forecast_metrics['avg_monthly_papers']}",
                        f"{forecast_metrics['months_to_clear']}",
                        f"{forecast_metrics['backlog_trend'].capitalize()}"
                    ]
                })
                st.dataframe(forecast_data, hide_index=True, use_container_width=True)
            else:
                st.info("Add more papers to see forecast metrics")

        # Research insights and recommendations section
        st.subheader("🧠 Research Insights & Recommendations")

        # Use direct insights generation instead of the potentially problematic function
        if analysis_data["total_papers"] > 0:
            # Create insights tabs
            insight_tabs = st.tabs(["Research Focus", "Reading Habits", "Research Planning"])

            with insight_tabs[0]:
                st.markdown("##### Research Focus Insights")
                focus_insights = []

                # Topic diversity insights
                if 'impact_metrics' in locals():
                    if impact_metrics["knowledge_breadth"] < 0.3 and analysis_data["total_papers"] > 5:
                        focus_insights.append(
                            "🔍 **Low topic diversity** - Consider exploring adjacent research areas to broaden your knowledge.")
                    elif impact_metrics["knowledge_breadth"] > 1.5:
                        focus_insights.append(
                            "🌐 **Very high topic diversity** - Consider focusing on fewer areas for deeper expertise.")

                    # Topic concentration insights
                    if impact_metrics["topic_concentration"] > 0.5 and analysis_data["topic_distribution"]:
                        top_topic = max(analysis_data["topic_distribution"].items(), key=lambda x: x[1])[0]
                        focus_insights.append(
                            f"🎯 **Strong specialization in '{top_topic}'** - Your focused approach may lead to expertise. Consider exploring how this topic connects to others.")

                if analysis_data["topic_distribution"]:
                    if len(analysis_data["topic_distribution"]) < 3 and analysis_data["total_papers"] > 5:
                        focus_insights.append(
                            "🔬 **Limited research domains** - Consider exploring related research areas to enhance interdisciplinary connections.")

                # Display insights
                if focus_insights:
                    for insight in focus_insights:
                        st.markdown(insight)
                        st.divider()
                else:
                    st.info("Add more papers with diverse topics to generate research focus insights.")

            with insight_tabs[1]:
                st.markdown("##### Reading Habits Insights")
                reading_insights = []

                # Reading progress insights
                completion_rate = analysis_data["completion_rate"]
                if completion_rate < 30:
                    reading_insights.append(
                        f"📉 **Low completion rate ({completion_rate:.1f}%)** - Focus on completing existing papers before adding new ones.")
                elif completion_rate > 70:
                    reading_insights.append(
                        f"📈 **High completion rate ({completion_rate:.1f}%)** - You're effectively completing papers. Consider increasing your research scope.")

                # Reading velocity insights
                if analysis_data["reading_velocity"] < 1:
                    reading_insights.append(
                        f"🐢 **Slow reading pace ({analysis_data['reading_velocity']:.1f} papers/month)** - This suggests deep analysis, which is good for thorough understanding.")
                elif analysis_data["reading_velocity"] > 5:
                    reading_insights.append(
                        f"🚀 **High reading velocity ({analysis_data['reading_velocity']:.1f} papers/month)** - Ensure you're retaining key information from your rapid reading.")

                # Reading consistency insights
                if 'impact_metrics' in locals():
                    if impact_metrics["reading_consistency"] < 0.3 and analysis_data["total_papers"] > 5:
                        reading_insights.append(
                            "📊 **Inconsistent reading pattern** - Develop a more consistent research schedule for better knowledge retention.")

                # Display insights
                if reading_insights:
                    for insight in reading_insights:
                        st.markdown(insight)
                        st.divider()
                else:
                    st.info("Continue adding and reading papers to generate reading habit insights.")

            with insight_tabs[2]:
                st.markdown("##### Research Planning Insights")
                planning_insights = []

                # Research velocity insights
                if 'impact_metrics' in locals():
                    if impact_metrics["research_velocity_trend"] == "decelerating" and analysis_data[
                        "total_papers"] > 5:
                        planning_insights.append(
                            "⏬ **Slowing research pace** - Set specific goals for paper discovery and reading to maintain momentum.")
                    elif impact_metrics["research_velocity_trend"] == "accelerating":
                        planning_insights.append(
                            "⏫ **Accelerating research velocity** - Ensure quality isn't sacrificed for quantity by maintaining thorough notes.")

                # Backlog insights
                if 'forecast_metrics' in locals():
                    if forecast_metrics["backlog_trend"] == "increasing" and forecast_metrics["current_backlog"] > 5:
                        planning_insights.append(
                            f"📚 **Growing backlog ({forecast_metrics['current_backlog']} papers)** - Consider allocating more time to clear your reading backlog.")

                    if forecast_metrics["months_to_clear"] != "∞" and float(forecast_metrics["months_to_clear"]) > 6:
                        planning_insights.append(
                            f"⏳ **Long backlog clearing time ({forecast_metrics['months_to_clear']} months)** - Consider being more selective with new papers or increasing reading rate.")

                # Display insights
                if planning_insights:
                    for insight in planning_insights:
                        st.markdown(insight)
                        st.divider()
                else:
                    st.info("Add more papers over time to generate research planning insights.")
        else:
            st.info("Add papers to receive research recommendations")


if __name__ == "__main__":
    main()