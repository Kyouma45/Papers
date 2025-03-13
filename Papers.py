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


def extract_arxiv_id(url):
    """Extract arXiv ID from URL or return the ID if directly provided."""
    # Match patterns like arxiv.org/abs/2312.12345 or arxiv.org/pdf/2312.12345.pdf
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)',
        r'(\d{4}\.\d{4,5})'  # Direct arXiv ID format
    ]

    for pattern in patterns:
        if match := re.search(pattern, url):
            return match.group(1)
    return None


def fetch_paper_details(arxiv_url):
    """Fetch paper details from arXiv API."""
    try:
        arxiv_id = extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            return None, "Invalid arXiv URL format"

        # Search for the paper
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])
        paper = next(client.results(search))

        # Create paper details dictionary
        paper_details = {
            'title': paper.title,
            'topics': [],
            'description': paper.summary.replace("\n", " ").strip(),
            'date': paper.published.strftime('%Y-%m-%d'),
            'link': arxiv_url
        }

        return paper_details, None
    except Exception as e:
        return None, f"Error fetching paper details: {str(e)}"


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
    except:
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
            colorbar=dict(
                thickness=15,
                title='Paper Count',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    # Color nodes by frequency
    node_adjacencies = []
    for node in nodes:
        count = sum(1 for topics in paper_df['Topics'] if isinstance(topics, list) and node in topics)
        node_adjacencies.append(count)

    node_trace.marker.color = node_adjacencies

    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Topic Relationship Network',
                        titlefont_size=16,
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

    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Paper", "📋 View Collection", "🔍 Search & Edit", "📊 Analysis"])

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
                    placeholder="Enter arXiv URL..."
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
                topics = st.text_input("🏷️ Topics",
                                       placeholder="AI, ML, NLP...",
                                       help="Comma-separated topics")

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
                    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
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
            filter_col1, filter_col2 = st.columns([1, 1])

            with filter_col1:
                # Topic filter with select all option
                all_topics = set(topic for topics_list in st.session_state.paper['Topics'] for topic in topics_list if
                                 isinstance(topics_list, list))
                topic_filter = st.multiselect("🏷️ Filter by Topics", options=["All"] + sorted(list(all_topics)))

                # Date range filter
                if not st.session_state.paper.empty and 'Date Added' in st.session_state.paper.columns:
                    min_date = pd.to_datetime(st.session_state.paper['Date Added']).min()
                    max_date = pd.to_datetime(st.session_state.paper['Date Added']).max()

                    if pd.notna(min_date) and pd.notna(max_date):
                        date_range = st.date_input(
                            "📅 Date Range",
                            value=(min_date.date(), max_date.date()),
                            min_value=min_date.date(),
                            max_value=max_date.date()
                        )
                    else:
                        date_range = None
                else:
                    date_range = None

            with filter_col2:
                # Reading status filter with multiselect
                status_options = ["Want to Read", "Reading", "Read"]
                status_filter = st.multiselect("📖 Reading Status",
                                               options=["All"] + status_options,
                                               default=["All"])

                # Text search filter
                text_filter = st.text_input("🔍 Search in Title/Description",
                                            placeholder="Enter keywords...")

        # View and sorting options
        view_col1, view_col2, view_col3 = st.columns([1, 1, 1])

        with view_col1:
            view_type = st.radio("👁️ View Type", ["Cards", "Compact List", "Detailed Table"], horizontal=True)

        with view_col2:
            sort_by = st.selectbox("🔄 Sort by", ["Date Added", "Title", "Reading Status", "Topics Count"])

        with view_col3:
            sort_direction = st.radio("⬆️⬇️ Order", ["Descending", "Ascending"], horizontal=True)

        # Apply filters
        papers_to_display = st.session_state.paper.copy()

        # Apply status filter
        if status_filter and "All" not in status_filter:
            papers_to_display = papers_to_display[papers_to_display['Reading Status'].isin(status_filter)]

        # Apply topic filter
        if topic_filter and "All" not in topic_filter:
            papers_to_display = papers_to_display[
                papers_to_display['Topics'].apply(lambda x: any(topic in x for topic in topic_filter))
            ]

        # Apply date range filter
        if date_range and len(date_range) == 2:
            papers_to_display['Date Added'] = pd.to_datetime(papers_to_display['Date Added'])
            start_date, end_date = date_range
            papers_to_display = papers_to_display[
                (papers_to_display['Date Added'].dt.date >= start_date) &
                (papers_to_display['Date Added'].dt.date <= end_date)
                ]

        # Apply text search filter
        if text_filter:
            papers_to_display = papers_to_display[
                papers_to_display['Title'].str.contains(text_filter, case=False, na=False) |
                papers_to_display['Description'].str.contains(text_filter, case=False, na=False)
                ]

        # Add topics count for sorting
        papers_to_display['Topics Count'] = papers_to_display['Topics'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

        # Apply sorting
        ascending = sort_direction == "Ascending"
        if sort_by == "Date Added":
            papers_to_display['Date Added'] = pd.to_datetime(papers_to_display['Date Added'])
            papers_to_display = papers_to_display.sort_values('Date Added', ascending=ascending)
        elif sort_by == "Title":
            papers_to_display = papers_to_display.sort_values('Title', ascending=ascending)
        elif sort_by == "Reading Status":
            # Custom sort order for Reading Status
            status_order = {"Read": 0, "Reading": 1, "Want to Read": 2}
            if ascending:
                status_order = {k: -v for k, v in status_order.items()}
            papers_to_display['Status Order'] = papers_to_display['Reading Status'].map(status_order)
            papers_to_display = papers_to_display.sort_values('Status Order', ascending=True)
        elif sort_by == "Topics Count":
            papers_to_display = papers_to_display.sort_values('Topics Count', ascending=ascending)

        # Display count of filtered papers
        st.markdown(f"**Showing {len(papers_to_display)} papers**")

        if papers_to_display.empty:
            st.info("No papers match your filter criteria.")
        else:
            # Display papers based on view type
            if view_type == "Cards":
                # Card view - grid layout
                paper_rows = [papers_to_display.iloc[i:i + 3] for i in range(0, len(papers_to_display), 3)]

                for row in paper_rows:
                    cols = st.columns(3)
                    for i, (_, paper) in enumerate(row.iterrows()):
                        with cols[i]:
                            status_color = {
                                "Want to Read": "blue",
                                "Reading": "orange",
                                "Read": "green"
                            }.get(paper['Reading Status'], "gray")

                            st.markdown(f"""
                            <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin:5px; height:200px; overflow:auto">
                                <h4>{paper['Title']}</h4>
                                <p><span style="color:{status_color}">●</span> {paper['Reading Status']}</p>
                                <p><small>Added: {pd.to_datetime(paper['Date Added']).strftime('%Y-%m-%d')}</small></p>
                                <p>Topics: {', '.join(paper['Topics']) if isinstance(paper['Topics'], list) else ''}</p>
                            </div>
                            """, unsafe_allow_html=True)

                            if paper['Link']:
                                st.markdown(f"[Open Paper]({paper['Link']})")

            elif view_type == "Compact List":
                # Compact list view
                for _, paper in papers_to_display.iterrows():
                    status_emoji = {
                        "Want to Read": "🔹",
                        "Reading": "🔶",
                        "Read": "✅"
                    }.get(paper['Reading Status'], "📄")

                    st.markdown(
                        f"{status_emoji} **{paper['Title']}** - {', '.join(paper['Topics']) if isinstance(paper['Topics'], list) else ''} *(Added: {pd.to_datetime(paper['Date Added']).strftime('%Y-%m-%d')})*")

                    col1, col2 = st.columns([1, 10])
                    with col1:
                        pass
                    with col2:
                        if paper['Link']:
                            st.markdown(f"[Open Paper]({paper['Link']})")

                    st.markdown("---")

            else:  # Detailed Table view
                # Convert to a format suitable for display
                display_df = papers_to_display.copy()
                display_df['Topics'] = display_df['Topics'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
                display_df['Date Added'] = pd.to_datetime(display_df['Date Added']).dt.strftime('%Y-%m-%d')

                # Reorder and select columns for display
                display_df = display_df[['Title', 'Reading Status', 'Date Added', 'Topics']]

                # Show as table
                st.dataframe(
                    display_df,
                    column_config={
                        "Title": st.column_config.TextColumn("Title"),
                        "Reading Status": st.column_config.TextColumn("Status"),
                        "Date Added": st.column_config.TextColumn("Added On"),
                        "Topics": st.column_config.TextColumn("Topics")
                    },
                    use_container_width=True,
                    hide_index=True
                )

        # Paper details section
        if len(papers_to_display) > 0:
            st.markdown("### 📑 Paper Details")
            paper_titles = ["Select a paper to view details"] + list(papers_to_display['Title'].values)
            selected_paper = st.selectbox("View paper details", options=paper_titles)

            if selected_paper != "Select a paper to view details":
                paper = papers_to_display[papers_to_display['Title'] == selected_paper].iloc[0]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(f"**Status:** {paper['Reading Status']}")
                    date_str = pd.to_datetime(paper['Date Added']).strftime('%Y-%m-%d')
                    st.markdown(f"**Added:** {date_str}")
                    if paper['Link']:
                        st.markdown(f"[Open Paper]({paper['Link']})")
                    st.markdown(
                        f"**Topics:** {', '.join(paper['Topics']) if isinstance(paper['Topics'], list) else ''}")

                with col2:
                    if paper['Description']:
                        st.markdown("### 📝 Notes")
                        st.markdown(paper['Description'])
                    else:
                        st.info("No notes added yet. Add some thoughts when you read the paper!")

    with tab3:
        st.markdown("### 🔍 Search & Edit Papers")
        search_term = st.text_input("Search papers by title, topic, or content...",
                                    placeholder="Type to search your collection...")

        if search_term:
            mask = (
                    st.session_state.paper['Title'].str.contains(search_term, case=False, na=False) |
                    st.session_state.paper['Topics'].apply(
                        lambda x: any(search_term.lower() in t.lower() for t in x if isinstance(x, list))) |
                    st.session_state.paper['Description'].str.contains(search_term, case=False, na=False)
            )
            search_results = st.session_state.paper[mask]

            if search_results.empty:
                st.warning("🔍 No papers found matching your search.")
            else:
                st.success(f"🎯 Found {len(search_results)} matching papers!")

                for index, paper in search_results.iterrows():
                    with st.expander(f"✏️ Edit: {paper['Title']}", expanded=False):
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
                                new_topics = st.text_input("Topics", value=", ".join(paper['Topics']))

                            new_description = st.text_area("Notes", value=paper['Description'], height=150)

                            # Add submit button for the form
                            submitted = st.form_submit_button("💾 Save Changes")
                            if submitted:
                                # Update the paper
                                st.session_state.paper.at[index, 'Title'] = new_title
                                st.session_state.paper.at[index, 'Reading Status'] = new_status
                                st.session_state.paper.at[index, 'Date Added'] = new_date.strftime('%Y-%m-%d')
                                st.session_state.paper.at[index, 'Link'] = new_link
                                st.session_state.paper.at[index, 'Topics'] = [t.strip() for t in new_topics.split(",") if
                                                                              t.strip()]
                                st.session_state.paper.at[index, 'Description'] = new_description
                                save_paper(st.session_state.paper)
                                st.success("✨ Changes saved successfully!")
                                st.rerun()

    with tab4:
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
                topics_per_paper = sum(
                    len(topics) for topics in st.session_state.paper['Topics'] if isinstance(topics, list)) / \
                                   analysis_data["total_papers"]
                unique_topics = len(analysis_data["topic_distribution"])
                topic_breadth = min(unique_topics / max(1, analysis_data["total_papers"]) * 100, 100)
                topic_depth = min(analysis_data["read_papers"] / max(1, analysis_data["total_papers"]) * 100, 100)

                metrics_data = pd.DataFrame({
                    'Metric': ['Topics per Paper', 'Unique Research Areas', 'Research Breadth', 'Research Depth'],
                    'Value': [f"{topics_per_paper:.1f}", unique_topics, f"{topic_breadth:.1f}%", f"{topic_depth:.1f}%"]
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
                st.markdown(f"**Research Momentum:** {momentum_text}")
            else:
                st.info("Add more papers to track research momentum")

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

        # Timeline with academic context
        st.subheader("📅 Research Timeline & Evolution")

        # Research evolution over time
        timeline_fig = create_reading_timeline(st.session_state.paper)
        if timeline_fig:
            timeline_fig.update_layout(title="Research Paper Timeline")
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("Add more papers with dates to visualize your research timeline")

        # Monthly activity with trend analysis
        if analysis_data["monthly_activity"] and len(analysis_data["monthly_activity"]) > 2:
            monthly_cols = st.columns([3, 1])
            with monthly_cols[0]:
                months = list(analysis_data["monthly_activity"].keys())
                counts = list(analysis_data["monthly_activity"].values())

                monthly_df = pd.DataFrame({
                    'Month': months,
                    'Papers Added': counts
                })
                monthly_df['Month'] = pd.to_datetime(monthly_df['Month'])
                monthly_df = monthly_df.sort_values('Month')

                # Add trendline
                fig = px.scatter(monthly_df, x='Month', y='Papers Added',
                                 trendline="rolling", trendline_options=dict(window=3),
                                 title="Research Intensity by Month")
                fig.update_traces(marker=dict(size=10))
                st.plotly_chart(fig, use_container_width=True)

            with monthly_cols[1]:
                st.markdown("##### Research Cycles")
                # Calculate moving average
                if len(monthly_df) >= 3:
                    monthly_df['Rolling Avg'] = monthly_df['Papers Added'].rolling(window=3).mean()
                    monthly_df = monthly_df.dropna()

                    # Calculate variance to identify consistency
                    variance = monthly_df['Papers Added'].var()
                    avg = monthly_df['Papers Added'].mean()
                    cov = (np.sqrt(variance) / avg) * 100 if avg > 0 else 0

                    cycle_metrics = pd.DataFrame({
                        'Metric': ['Average Monthly Papers', 'Research Consistency', 'Peak Research Month'],
                        'Value': [
                            f"{avg:.1f}",
                            f"{100 - min(cov, 100):.0f}% (lower variance = more consistent)",
                            monthly_df.loc[monthly_df['Papers Added'].idxmax(), 'Month'].strftime('%b %Y')
                        ]
                    })
                    st.dataframe(cycle_metrics, hide_index=True, use_container_width=True)
                else:
                    st.info("More data needed for cycle analysis")
        else:
            st.info("Add more papers over time to see research activity patterns")

        # Research insights and recommendations
        st.subheader("🧠 Research Insights & Recommendations")
        insight_cols = st.columns(2)

        with insight_cols[0]:
            st.markdown("##### Current Research Patterns")

            # Generate insights based on the data
            insights = []

            if analysis_data["total_papers"] > 0:
                # Topic concentration
                top_topic = max(analysis_data["topic_distribution"].items(), key=lambda x: x[1])[0] if analysis_data[
                    "topic_distribution"] else "None"
                top_topic_percent = (
                            max(analysis_data["topic_distribution"].values()) / analysis_data["total_papers"] * 100) if \
                analysis_data["topic_distribution"] else 0

                if top_topic_percent > 50:
                    insights.append(f"Strong focus on '{top_topic}' ({top_topic_percent:.1f}% of papers)")
                elif top_topic_percent > 30:
                    insights.append(f"Moderate specialization in '{top_topic}' ({top_topic_percent:.1f}% of papers)")

                # Reading completion
                completion_rate = analysis_data["completion_rate"]
                if completion_rate < 30:
                    insights.append(
                        f"Low completion rate ({completion_rate:.1f}%) - consider focusing on current papers")
                elif completion_rate > 70:
                    insights.append(
                        f"High completion rate ({completion_rate:.1f}%) - good follow-through on selected papers")

                # Reading velocity
                if analysis_data["reading_velocity"] < 1:
                    insights.append(
                        f"Reading pace ({analysis_data['reading_velocity']:.1f} papers/month) suggests deep analysis")
                elif analysis_data["reading_velocity"] > 5:
                    insights.append(
                        f"High reading velocity ({analysis_data['reading_velocity']:.1f} papers/month) indicates survey research")

            if not insights:
                insights = ["Add more papers to generate research insights"]

            for i, insight in enumerate(insights):
                st.markdown(f"**{i + 1}.** {insight}")

        with insight_cols[1]:
            st.markdown("##### Research Recommendations")

            # Generate recommendations based on the analysis
            recommendations = []

            if analysis_data["total_papers"] > 0:
                # Topic exploration
                if len(analysis_data["topic_distribution"]) < 3 and analysis_data["total_papers"] > 5:
                    recommendations.append("Consider exploring more diverse research areas")

                # Reading focus
                if analysis_data["want_to_read_papers"] > analysis_data["read_papers"] * 2:
                    recommendations.append("Focus on reading existing papers before adding more")

                # Recent topics
                if analysis_data["recent_topics"]:
                    recommendations.append(f"Recent focus on: {', '.join(analysis_data['recent_topics'][:3])}")

                # Topic connections
                if len(analysis_data["topic_distribution"]) > 3:
                    recommendations.append("Look for interdisciplinary connections between your research domains")

            if not recommendations:
                recommendations = ["Add more papers to receive research recommendations"]

            for i, recommendation in enumerate(recommendations):
                st.markdown(f"**{i + 1}.** {recommendation}")

if __name__ == "__main__":
    main()
