import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime, timedelta
import random
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from collections import Counter, defaultdict
import requests
import urllib3
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

# Optional advanced analytics imports
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False

try:
    import arxiv
    HAS_ARXIV = True
except ImportError:
    HAS_ARXIV = False

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
            print(f"❌ Invalid arXiv URL format: {arxiv_url}")
            return None, "Invalid arXiv URL format"
        
        print(f"📡 Fetching paper details for arXiv ID: {arxiv_id}")
        
        # Make direct request to arXiv API with SSL verification disabled
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        print(f"🌐 Making API request to: {api_url}")
        
        try:
            response = requests.get(api_url, verify=False, timeout=10)
            print(f"📊 API Response Status: {response.status_code}")
            print(f"📋 Content Type: {response.headers.get('content-type', 'N/A')}")
            
            if response.status_code != 200:
                error_msg = f"API returned status code {response.status_code}"
                print(f"❌ {error_msg}")
                return None, error_msg
            
            # XML namespaces for arXiv API
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Parse XML response
            root = ET.fromstring(response.content)
            print(f"✅ Successfully parsed XML response")
            
            # Check if paper exists
            entry = root.find('.//atom:entry', ns)
            if entry is None:
                error_msg = f"No paper found with ID {arxiv_id}"
                print(f"❌ {error_msg}")
                return None, error_msg
            
            print(f"📄 Found paper entry, extracting details...")
            
            # Extract basic paper details
            title_elem = entry.find('./atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None else "Unknown Title"
            print(f"📖 Title: {title}")
            
            summary_elem = entry.find('./atom:summary', ns)
            summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None else "No summary available"
            print(f"📝 Summary length: {len(summary)} characters")
            
            # Get publication and update dates
            published_elem = entry.find('./atom:published', ns)
            published = published_elem.text if published_elem is not None else None
            published_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d') if published else ""
            print(f"📅 Published: {published_date}")
            
            updated_elem = entry.find('./atom:updated', ns)
            updated = updated_elem.text if updated_elem is not None else None
            updated_date = datetime.strptime(updated, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d') if updated else ""
            if updated_date and updated_date != published_date:
                print(f"🔄 Last Updated: {updated_date}")
            
            # Extract authors
            authors = []
            author_elements = entry.findall('./atom:author', ns)
            print(f"👥 Found {len(author_elements)} author(s):")
            for author_elem in author_elements:
                name_elem = author_elem.find('./atom:name', ns)
                if name_elem is not None:
                    author_name = name_elem.text.strip()
                    authors.append(author_name)
                    print(f"   📝 {author_name}")
            
            # Extract comment (additional metadata)
            comment_elem = entry.find('./arxiv:comment', ns)
            comment = comment_elem.text.strip() if comment_elem is not None else ""
            if comment:
                print(f"💬 Comment: {comment}")
            
            # Extract PDF link
            pdf_link = ""
            link_elements = entry.findall('./atom:link', ns)
            for link_elem in link_elements:
                if link_elem.get('type') == 'application/pdf':
                    pdf_link = link_elem.get('href', '')
                    print(f"📑 PDF Link: {pdf_link}")
                    break
            
            # Extract primary category
            primary_category_elem = entry.find('./arxiv:primary_category', ns)
            primary_category = primary_category_elem.get('term', '') if primary_category_elem is not None else ""
            if primary_category:
                print(f"🏷️ Primary Category: {primary_category}")
            
            # Extract all categories (including subcategories)
            all_categories = []
            categories = []  # For backward compatibility (primary categories only)
            category_elements = entry.findall('./atom:category', ns)
            print(f"📂 Found {len(category_elements)} categor{'ies' if len(category_elements) != 1 else 'y'}:")
            
            for category_elem in category_elements:
                term = category_elem.get('term', '')
                all_categories.append(term)
                print(f"   🏷️ {term}")
                
                # Extract primary category for backward compatibility
                if term and '.' in term:
                    primary = term.split('.')[0]
                    if primary not in categories:
                        categories.append(primary)
            
            # Extract version information from ID
            id_elem = entry.find('./atom:id', ns)
            version = ""
            if id_elem is not None:
                id_text = id_elem.text
                # Extract version from ID like "http://arxiv.org/abs/2301.07041v2"
                version_match = re.search(r'v(\d+)$', id_text)
                if version_match:
                    version = version_match.group(1)
                    print(f"📌 Version: v{version}")
            
            print(f"✅ Successfully extracted all paper details")
            
            # Create enhanced paper details dictionary
            paper_details = {
                'title': title,
                'authors': authors,
                'topics': categories,  # Primary categories for backward compatibility
                'all_categories': all_categories,  # All ArXiv categories including subcategories
                'primary_category': primary_category,
                'description': summary,
                'comment': comment,
                'version': version,
                'date_added': datetime.today().strftime('%Y-%m-%d'),  # Current date when added
                'date_published': published_date,  # ArXiv publication date
                'date_updated': updated_date,  # ArXiv last update date
                'link': arxiv_url,
                'pdf_link': pdf_link
            }
            
            print(f"📊 Paper details summary:")
            print(f"   📖 Title: {title[:60]}{'...' if len(title) > 60 else ''}")
            print(f"   👥 Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
            print(f"   🏷️ Categories: {', '.join(categories)}")
            print(f"   📅 Published: {published_date}")
            if updated_date and updated_date != published_date:
                print(f"   🔄 Updated: {updated_date}")
            
            return paper_details, None
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg
            
    except Exception as e:
        error_msg = f"Error fetching paper details: {str(e)}"
        print(f"❌ {error_msg}")
        return None, error_msg


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


def is_arxiv_url(url):
    """Check if URL is from ArXiv."""
    if not url:
        return False
    return 'arxiv.org' in str(url).lower()


def needs_paper_update(paper):
    """Check if a paper needs updating based on missing fields."""
    missing_fields = []
    
    # Check for missing fields that can be extracted from ArXiv
    if not paper.get('Authors') or (isinstance(paper.get('Authors'), list) and len(paper.get('Authors')) == 0):
        missing_fields.append('Authors')
    
    if not paper.get('Date Updated'):
        missing_fields.append('Date Updated')
    
    if not paper.get('Comment'):
        missing_fields.append('Comment')
    
    if not paper.get('PDF Link'):
        missing_fields.append('PDF Link')
    
    if not paper.get('Primary Category'):
        missing_fields.append('Primary Category')
    
    if not paper.get('All Categories') or (isinstance(paper.get('All Categories'), list) and len(paper.get('All Categories')) == 0):
        missing_fields.append('All Categories')
    
    if not paper.get('Version'):
        missing_fields.append('Version')
    
    # Only check Date Published for ArXiv papers
    if not paper.get('Date Published') and is_arxiv_url(paper.get('Link')):
        missing_fields.append('Date Published')
    
    return len(missing_fields) > 0, missing_fields


def auto_fill_missing_paper_info():
    """Auto-fill missing information for all ArXiv papers in the database."""
    if 'paper' not in st.session_state or st.session_state.paper.empty:
        st.error("No papers found in the database!")
        return 0, 0
    
    paper_df = st.session_state.paper.copy()
    
    # Find papers that need updating
    papers_to_update = []
    for index, paper in paper_df.iterrows():
        if is_arxiv_url(paper.get('Link')):
            needs_upd, missing_fields = needs_paper_update(paper)
            if needs_upd:
                papers_to_update.append((index, paper, missing_fields))
    
    if not papers_to_update:
        st.success("✅ All ArXiv papers already have complete information!")
        return 0, 0
    
    st.info(f"🔍 Found {len(papers_to_update)} ArXiv papers that need updating")
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    updated_count = 0
    failed_count = 0
    
    for i, (index, paper, missing_fields) in enumerate(papers_to_update):
        # Update progress
        progress = (i + 1) / len(papers_to_update)
        progress_bar.progress(progress)
        status_text.text(f"Processing {i+1}/{len(papers_to_update)}: {paper['Title'][:50]}...")
        
        # Fetch enhanced details from ArXiv
        enhanced_data, error = fetch_paper_details(paper['Link'])
        
        if error:
            failed_count += 1
            continue
        
        # Update paper fields with new information
        updated_fields = []
        
        # Update Authors
        if ('Authors' in missing_fields and enhanced_data.get('authors')):
            paper_df.at[index, 'Authors'] = enhanced_data['authors']
            updated_fields.append('Authors')
        
        # Update Date Updated
        if ('Date Updated' in missing_fields and enhanced_data.get('date_updated')):
            paper_df.at[index, 'Date Updated'] = enhanced_data['date_updated']
            updated_fields.append('Date Updated')
        
        # Update Comment
        if ('Comment' in missing_fields and enhanced_data.get('comment')):
            paper_df.at[index, 'Comment'] = enhanced_data['comment']
            updated_fields.append('Comment')
        
        # Update PDF Link
        if ('PDF Link' in missing_fields and enhanced_data.get('pdf_link')):
            paper_df.at[index, 'PDF Link'] = enhanced_data['pdf_link']
            updated_fields.append('PDF Link')
        
        # Update Primary Category
        if ('Primary Category' in missing_fields and enhanced_data.get('primary_category')):
            paper_df.at[index, 'Primary Category'] = enhanced_data['primary_category']
            updated_fields.append('Primary Category')
        
        # Update All Categories
        if ('All Categories' in missing_fields and enhanced_data.get('all_categories')):
            paper_df.at[index, 'All Categories'] = enhanced_data['all_categories']
            updated_fields.append('All Categories')
        
        # Update Version
        if ('Version' in missing_fields and enhanced_data.get('version')):
            paper_df.at[index, 'Version'] = enhanced_data['version']
            updated_fields.append('Version')
        
        # Update Date Published
        if ('Date Published' in missing_fields and enhanced_data.get('date_published')):
            paper_df.at[index, 'Date Published'] = enhanced_data['date_published']
            updated_fields.append('Date Published')
        
        if updated_fields:
            updated_count += 1
        
        # Small delay to be respectful to the API
        import time
        time.sleep(0.5)
    
    # Update the session state and save
    st.session_state.paper = paper_df
    save_paper(paper_df)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    st.success("✅ Auto-fill completed!")
    st.info(f"📊 Successfully updated {updated_count} papers")
    if failed_count > 0:
        st.warning(f"⚠️ Failed to update {failed_count} papers")
    
    return updated_count, failed_count


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
            "research_velocity_trend": "stable",
            "arxiv_engagement": 0,
            "category_diversity": 0,
            "version_awareness": 0,
            "publication_recency": 0
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

    # Calculate reading consistency - should be based on Date Read for actual reading pattern
    reading_consistency = 0
    if 'Date Read' in paper_df.columns:
        # Use Date Read for consistency calculation (actual reading pattern)
        read_papers_df = paper_df[(paper_df['Reading Status'] == 'Read') & paper_df['Date Read'].notna()]
        if len(read_papers_df) > 1:
            paper_df['Date Read'] = pd.to_datetime(paper_df['Date Read'], errors='coerce')
            read_papers_df['Month'] = read_papers_df['Date Read'].dt.strftime('%Y-%m')
            monthly_read_counts = read_papers_df.groupby('Month').size()
            
            if len(monthly_read_counts) > 1 and monthly_read_counts.mean() > 0:
                reading_consistency = 1 - (monthly_read_counts.std() / monthly_read_counts.mean())
                reading_consistency = max(0, min(1, reading_consistency))  # Ensure between 0 and 1
    
    # Fall back to Date Added if Date Read not available
    if reading_consistency == 0:
        paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
        paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')
        monthly_counts = paper_df.groupby('Month').size()

        if len(monthly_counts) > 1:
            reading_consistency = 1 - (monthly_counts.std() / monthly_counts.mean() if monthly_counts.mean() > 0 else 0)
            reading_consistency = max(0, min(1, reading_consistency))  # Ensure between 0 and 1

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

    # NEW METRIC: ArXiv Engagement (percentage of papers from ArXiv)
    arxiv_engagement = 0
    if 'PDF Link' in paper_df.columns or 'Primary Category' in paper_df.columns:
        arxiv_papers = 0
        for idx, row in paper_df.iterrows():
            # Check if it's an ArXiv paper by PDF link or Primary Category
            is_arxiv = (
                ('PDF Link' in row and pd.notna(row['PDF Link']) and 'arxiv.org' in str(row['PDF Link'])) or
                ('Primary Category' in row and pd.notna(row['Primary Category']) and str(row['Primary Category']).strip())
            )
            if is_arxiv:
                arxiv_papers += 1
        arxiv_engagement = arxiv_papers / max(1, total_papers)

    # NEW METRIC: Category Diversity (using Primary Category and All Categories)
    category_diversity = 0
    if 'Primary Category' in paper_df.columns:
        primary_cats = set()
        for cat in paper_df['Primary Category']:
            if pd.notna(cat) and str(cat).strip():
                primary_cats.add(str(cat).strip())
        category_diversity = len(primary_cats) / max(1, total_papers)
    
    # Enhance with All Categories if available
    if 'All Categories' in paper_df.columns:
        all_cats = set()
        for cats in paper_df['All Categories']:
            if isinstance(cats, list):
                all_cats.update(cats)
            elif pd.notna(cats) and str(cats).strip():
                cat_list = [c.strip() for c in str(cats).split(';')]
                all_cats.update(cat_list)
        if all_cats:
            category_diversity = max(category_diversity, len(all_cats) / max(1, total_papers))

    # NEW METRIC: Version Awareness (how often you read updated versions)
    version_awareness = 0
    if 'Version' in paper_df.columns:
        versioned_papers = 0
        for version in paper_df['Version']:
            if pd.notna(version) and str(version).strip():
                import re
                # Check if version is v2 or higher (indicating you read updated papers)
                match = re.search(r'v(\d+)', str(version))
                if match and int(match.group(1)) > 1:
                    versioned_papers += 1
        version_awareness = versioned_papers / max(1, total_papers)

    # NEW METRIC: Publication Recency (how recent are the papers you're reading)
    publication_recency = 0
    if 'Date Published' in paper_df.columns:
        current_date = pd.Timestamp.now()
        recent_papers = 0
        total_dated_papers = 0
        
        for date_pub in paper_df['Date Published']:
            if pd.notna(date_pub):
                pub_date = pd.to_datetime(date_pub, errors='coerce')
                if pd.notna(pub_date):
                    total_dated_papers += 1
                    # Consider papers from last 2 years as "recent"
                    if (current_date - pub_date).days <= 730:
                        recent_papers += 1
        
        if total_dated_papers > 0:
            publication_recency = recent_papers / total_dated_papers

    # Determine research velocity trend - should use Date Read for actual reading trend
    velocity_trend = "stable"
    if 'Date Read' in paper_df.columns:
        # Use Date Read for velocity trend (actual reading pattern)
        read_papers_df = paper_df[(paper_df['Reading Status'] == 'Read') & paper_df['Date Read'].notna()]
        if len(read_papers_df) > 0:
            paper_df['Date Read'] = pd.to_datetime(paper_df['Date Read'], errors='coerce')
            read_papers_df['Month'] = read_papers_df['Date Read'].dt.strftime('%Y-%m')
            monthly_read_counts = read_papers_df.groupby('Month').size()
            
            if len(monthly_read_counts) >= 3:
                recent_months = sorted(monthly_read_counts.index)[-3:]
                if len(recent_months) == 3:
                    start_count = monthly_read_counts[recent_months[0]]
                    end_count = monthly_read_counts[recent_months[2]]

                    if end_count > start_count * 1.2:
                        velocity_trend = "accelerating"
                    elif end_count < start_count * 0.8:
                        velocity_trend = "decelerating"
                    else:
                        velocity_trend = "stable"
    
    # Fall back to Date Added if no Date Read data
    if velocity_trend == "stable" and len(paper_df) > 0:
        paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
        paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')
        monthly_counts = paper_df.groupby('Month').size()
        
        if len(monthly_counts) >= 3:
            recent_months = sorted(monthly_counts.index)[-3:]
            if len(recent_months) == 3:
                start_count = monthly_counts[recent_months[0]]
                end_count = monthly_counts[recent_months[2]]

                if end_count > start_count * 1.2:
                    velocity_trend = "accelerating"
                elif end_count < start_count * 0.8:
                    velocity_trend = "decelerating"

    return {
        "knowledge_breadth": round(knowledge_breadth, 2),
        "knowledge_depth": round(knowledge_depth, 2),
        "research_efficiency": round(reading_efficiency, 2),
        "topic_concentration": round(topic_concentration, 2),
        "reading_consistency": round(reading_consistency, 2),
        "exploration_vs_exploitation": round(exploration_ratio, 2),
        "research_velocity_trend": velocity_trend,
        "arxiv_engagement": round(arxiv_engagement, 2),
        "category_diversity": round(category_diversity, 2),
        "version_awareness": round(version_awareness, 2),
        "publication_recency": round(publication_recency, 2)
    }


def create_radar_chart(metrics):
    """Create a radar chart of research metrics."""
    categories = ['Knowledge Breadth', 'Knowledge Depth', 'Research Efficiency',
                  'Reading Consistency', 'Topic Focus', 'Exploration', 
                  'ArXiv Engagement', 'Category Diversity', 'Version Awareness', 'Publication Recency']

    values = [
        metrics["knowledge_breadth"] * 5,  # Scale to 0-5
        metrics["knowledge_depth"] / 2,  # Scale down if very high
        metrics["research_efficiency"] * 5,
        metrics["reading_consistency"] * 5,
        metrics["topic_concentration"] * 5,
        metrics["exploration_vs_exploitation"] * 5,
        metrics["arxiv_engagement"] * 5,  # New metric
        metrics["category_diversity"] * 5,  # New metric
        metrics["version_awareness"] * 5,  # New metric
        metrics["publication_recency"] * 5  # New metric
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
        # First try normal loading with UTF-8 encoding
        if os.path.exists(PAPER_FILE):
            with open(PAPER_FILE, 'r', encoding='utf-8') as f:
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
                'Title', 'Reading Status', 'Date Added', 'Date Published', 'Date Read', 'Date Updated', 
                'Link', 'PDF Link', 'Topics', 'All Categories', 'Primary Category', 'Authors', 
                'Description', 'Comment', 'Version'
            ])

        # Ensure all required columns exist
        required_columns = {
            'Title': str,
            'Reading Status': str,
            'Date Added': str,
            'Date Published': str,  # ArXiv publication date
            'Date Read': str,       # Date when user finished reading
            'Date Updated': str,    # ArXiv last update date
            'Link': str,
            'PDF Link': str,        # Direct PDF link
            'Topics': list,         # Primary categories (backward compatibility)
            'All Categories': list, # All ArXiv categories including subcategories
            'Primary Category': str,# Primary ArXiv category
            'Authors': list,        # List of authors
            'Description': str,
            'Comment': str,         # ArXiv comment field
            'Version': str          # ArXiv version (v1, v2, etc.)
        }

        for col, dtype in required_columns.items():
            if col not in paper_df.columns:
                if dtype == list:
                    paper_df[col] = [[] for _ in range(len(paper_df))]
                elif col in ['Date Published', 'Date Read', 'Date Updated', 'PDF Link', 'Primary Category', 'Comment', 'Version']:
                    paper_df[col] = ''  # Empty string for optional fields
                else:
                    paper_df[col] = dtype()

        # Clean up list columns
        list_columns = ['Topics', 'All Categories', 'Authors']
        for col in list_columns:
            if col in paper_df.columns:
                paper_df[col] = paper_df[col].apply(
                    lambda x: [t.strip() for t in (x if isinstance(x, list) else
                                                   (x.split(',') if isinstance(x, str) and x else []))]
                )

        # Ensure Date Added is in correct format
        paper_df['Date Added'] = paper_df['Date Added'].apply(
            lambda x: datetime.today().strftime('%Y-%m-%d') if pd.isna(x) or x == '' else str(x)
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
        # Check if it's a NaT (Not a Time) value
        if pd.isna(obj):
            return None
        return obj.strftime('%Y-%m-%d')
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def format_date_for_storage(date_obj):
    """Format date object for storage, handling various date input types."""
    if date_obj is None or pd.isna(date_obj):
        return ''
    
    # Handle single date objects
    if hasattr(date_obj, 'strftime'):
        try:
            return date_obj.strftime('%Y-%m-%d')
        except:
            return ''  # Return empty string for NaT or invalid dates
    
    # Handle date strings
    if isinstance(date_obj, str):
        return date_obj if date_obj else ''
    
    # Convert to string as fallback
    return str(date_obj) if date_obj else ''


def safe_date_parse(date_str):
    """Safely parse date strings, returning None for NaT values."""
    try:
        if pd.isna(date_str) or date_str == '' or date_str is None:
            return None
        parsed_date = pd.to_datetime(date_str)
        if pd.isna(parsed_date):
            return None
        # Convert to Python date object for Streamlit
        return parsed_date.date() if hasattr(parsed_date, 'date') else None
    except Exception as e:
        return None
    """Safely parse date strings, returning None for NaT values."""
    try:
        date = pd.to_datetime(date_str)
        return date.date() if pd.notnull(date) else datetime.today().date()
    except Exception as e:
        return datetime.today().date()


def save_paper(paper_df):
    # Create a copy to avoid modifying the original DataFrame
    paper_df = paper_df.copy()
    
    # Handle all date columns properly
    date_columns = ['Date Added', 'Date Published', 'Date Read']
    for col in date_columns:
        if col in paper_df.columns:
            paper_df[col] = paper_df[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if pd.notna(x) and isinstance(x, (datetime, pd.Timestamp)) 
                else '' if pd.isna(x) else str(x)
            )
    
    paper_dict = paper_df.to_dict('records')
    with open(PAPER_FILE, 'w', encoding='utf-8') as f:
        json.dump(paper_dict, f, default=serialize_dates, ensure_ascii=False, indent=2)


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


def add_paper(title, reading_status, date_added, link, topics, description, date_published='', date_read='', 
              authors=None, all_categories=None, primary_category='', pdf_link='', comment='', 
              version='', date_updated=''):
    # Check if paper with same title exists (case-insensitive)
    existing_titles = st.session_state.paper['Title'].str.lower()
    if title.lower() in existing_titles.values:
        st.error(f"🚫 A paper with title '{title}' already exists!")
        return False

    # If title doesn't exist, proceed with adding the paper
    topics_list = topics if isinstance(topics, list) else []
    authors_list = authors if isinstance(authors, list) else []
    all_categories_list = all_categories if isinstance(all_categories, list) else []
    
    # Ensure dates are in proper string format
    date_added = format_date_for_storage(date_added)
    
    # Set Date Published to Date Added if not provided (they should be the same)
    if not date_published:
        date_published = date_added
    else:
        date_published = format_date_for_storage(date_published)
    
    # Format update date if provided
    date_updated = format_date_for_storage(date_updated) if date_updated else ''
    
    # Automatically set Date Read to today if status is "Read"
    if reading_status == 'Read':
        if not date_read:
            date_read = datetime.today().strftime('%Y-%m-%d')
        else:
            date_read = format_date_for_storage(date_read)
    else:
        date_read = ''  # Clear date read if status is not "Read"
    
    new_paper = pd.DataFrame({
        'Title': [title],
        'Reading Status': [reading_status],
        'Date Added': [date_added],
        'Date Published': [date_published],
        'Date Read': [date_read],
        'Date Updated': [date_updated],
        'Link': [link],
        'PDF Link': [pdf_link],
        'Topics': [topics_list],
        'All Categories': [all_categories_list],
        'Primary Category': [primary_category],
        'Authors': [authors_list],
        'Description': [description],
        'Comment': [comment],
        'Version': [version]
    })
    st.session_state.paper = pd.concat([st.session_state.paper, new_paper], ignore_index=True)
    st.session_state.paper = st.session_state.paper.sort_values('Title')
    save_paper(st.session_state.paper)
    return True


def edit_paper(index, new_title, new_status, new_date, new_link, new_topics, new_description, 
               new_date_published='', new_date_read=''):
    # Convert topics string to list
    topics_list = [t.strip() for t in new_topics.split(",") if t.strip()]
    
    # Ensure dates are in proper string format
    new_date = format_date_for_storage(new_date)
    
    # Set Date Published to Date Added if not provided (they should be the same)
    if not new_date_published:
        new_date_published = new_date
    else:
        new_date_published = format_date_for_storage(new_date_published)
    
    # Automatically set Date Read to today if status changes to "Read" and no date is provided
    if new_status == 'Read':
        if not new_date_read:
            new_date_read = datetime.today().strftime('%Y-%m-%d')
        else:
            new_date_read = format_date_for_storage(new_date_read)
    else:
        new_date_read = ''  # Clear date read if status is not "Read"
    
    st.session_state.paper.at[index, 'Title'] = new_title
    st.session_state.paper.at[index, 'Reading Status'] = new_status
    st.session_state.paper.at[index, 'Date Added'] = new_date
    st.session_state.paper.at[index, 'Date Published'] = new_date_published
    st.session_state.paper.at[index, 'Date Read'] = new_date_read
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
            "recent_topics": [],
            "author_engagement": {},
            "category_distribution": {},
            "version_distribution": {},
            "publication_timeline": {},
            "arxiv_statistics": {}
        }

    # Ensure date format is consistent for both Date Added and Date Read
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
    if 'Date Read' in paper_df.columns:
        paper_df['Date Read'] = pd.to_datetime(paper_df['Date Read'], errors='coerce')
    if 'Date Published' in paper_df.columns:
        paper_df['Date Published'] = pd.to_datetime(paper_df['Date Published'], errors='coerce')
    if 'Date Updated' in paper_df.columns:
        paper_df['Date Updated'] = pd.to_datetime(paper_df['Date Updated'], errors='coerce')

    # Basic counts
    total_papers = len(paper_df)
    read_papers = len(paper_df[paper_df['Reading Status'] == 'Read'])
    reading_papers = len(paper_df[paper_df['Reading Status'] == 'Reading'])
    want_to_read_papers = len(paper_df[paper_df['Reading Status'] == 'Want to Read'])

    # Reading velocity should be based on Date Read for completed papers
    reading_velocity = 0
    if read_papers > 0 and 'Date Read' in paper_df.columns:
        read_papers_df = paper_df[(paper_df['Reading Status'] == 'Read') & paper_df['Date Read'].notna()]
        if len(read_papers_df) > 0:
            earliest_read = read_papers_df['Date Read'].min()
            latest_read = read_papers_df['Date Read'].max()
            
            if pd.notnull(earliest_read) and pd.notnull(latest_read):
                months_diff = (latest_read.year - earliest_read.year) * 12 + (latest_read.month - earliest_read.month)
                months_diff = max(1, months_diff)  # Avoid division by zero
                reading_velocity = len(read_papers_df) / months_diff
    
    # If no Date Read data, fall back to Date Added for acquisition velocity
    if reading_velocity == 0 and total_papers > 0:
        earliest_date = paper_df['Date Added'].min()
        latest_date = paper_df['Date Added'].max()

        if pd.notnull(earliest_date) and pd.notnull(latest_date):
            months_diff = (latest_date.year - earliest_date.year) * 12 + (latest_date.month - earliest_date.month)
            months_diff = max(1, months_diff)
            reading_velocity = total_papers / months_diff

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

    # NEW: Author engagement analysis
    author_engagement = {}
    if 'Authors' in paper_df.columns:
        author_counts = Counter()
        for authors in paper_df['Authors']:
            if isinstance(authors, list):
                for author in authors:
                    author_counts[author] += 1
            elif isinstance(authors, str) and authors.strip():
                author_list = [author.strip() for author in authors.split(',')]
                for author in author_list:
                    author_counts[author] += 1
        
        author_engagement = {
            "most_read_authors": dict(author_counts.most_common(10)),
            "unique_authors": len(author_counts),
            "avg_authors_per_paper": sum(author_counts.values()) / max(1, total_papers)
        }

    # NEW: Category distribution analysis
    category_distribution = {}
    if 'Primary Category' in paper_df.columns:
        primary_cat_counts = Counter()
        for cat in paper_df['Primary Category']:
            if pd.notna(cat) and str(cat).strip():
                primary_cat_counts[str(cat).strip()] += 1
        
        category_distribution["primary_categories"] = dict(primary_cat_counts.most_common())
    
    if 'All Categories' in paper_df.columns:
        all_cat_counts = Counter()
        for cats in paper_df['All Categories']:
            if isinstance(cats, list):
                for cat in cats:
                    all_cat_counts[cat] += 1
            elif pd.notna(cats) and str(cats).strip():
                cat_list = [c.strip() for c in str(cats).split(';')]
                for cat in cat_list:
                    all_cat_counts[cat] += 1
        
        category_distribution["all_categories"] = dict(all_cat_counts.most_common(15))

    # NEW: Version distribution analysis
    version_distribution = {}
    if 'Version' in paper_df.columns:
        version_counts = Counter()
        for version in paper_df['Version']:
            if pd.notna(version) and str(version).strip():
                version_counts[str(version)] += 1
        
        # Calculate updated papers percentage
        updated_papers = sum(1 for v in version_counts.keys() 
                           if v and any(char.isdigit() and int(char) > 1 for char in v))
        
        version_distribution = {
            "version_counts": dict(version_counts.most_common()),
            "updated_papers_percentage": round((updated_papers / max(1, total_papers)) * 100, 1)
        }

    # NEW: Publication timeline analysis
    publication_timeline = {}
    if 'Date Published' in paper_df.columns:
        pub_papers = paper_df[paper_df['Date Published'].notna()]
        if len(pub_papers) > 0:
            pub_papers['Pub Year'] = pub_papers['Date Published'].dt.year
            yearly_pubs = pub_papers.groupby('Pub Year').size()
            
            publication_timeline = {
                "papers_by_year": yearly_pubs.to_dict(),
                "earliest_paper": int(yearly_pubs.index.min()) if len(yearly_pubs) > 0 else None,
                "latest_paper": int(yearly_pubs.index.max()) if len(yearly_pubs) > 0 else None,
                "avg_paper_age_years": round((pd.Timestamp.now() - pub_papers['Date Published'].mean()).days / 365.25, 1)
            }

    # NEW: ArXiv-specific statistics
    arxiv_statistics = {}
    arxiv_papers = 0
    pdf_available = 0
    
    # Count ArXiv papers and PDFs
    for idx, row in paper_df.iterrows():
        is_arxiv = (
            ('PDF Link' in row and pd.notna(row['PDF Link']) and 'arxiv.org' in str(row['PDF Link'])) or
            ('Primary Category' in row and pd.notna(row['Primary Category']) and str(row['Primary Category']).strip())
        )
        if is_arxiv:
            arxiv_papers += 1
        
        if 'PDF Link' in row and pd.notna(row['PDF Link']) and str(row['PDF Link']).strip():
            pdf_available += 1
    
    arxiv_statistics = {
        "arxiv_papers": arxiv_papers,
        "arxiv_percentage": round((arxiv_papers / max(1, total_papers)) * 100, 1),
        "pdf_available": pdf_available,
        "pdf_percentage": round((pdf_available / max(1, total_papers)) * 100, 1)
    }

    # Monthly activity should reflect actual reading when possible
    if 'Date Read' in paper_df.columns:
        # Use Date Read for completed papers
        read_papers_df = paper_df[(paper_df['Reading Status'] == 'Read') & paper_df['Date Read'].notna()]
        if len(read_papers_df) > 0:
            read_papers_df['Month'] = read_papers_df['Date Read'].dt.strftime('%Y-%m')
            monthly_reading = read_papers_df.groupby('Month').size()
            # Also include papers added (acquisition)
            paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')
            monthly_added = paper_df.groupby('Month').size()
            # Combine both metrics
            monthly_activity = {
                "reading_activity": monthly_reading.to_dict(),
                "acquisition_activity": monthly_added.to_dict()
            }
        else:
            # Fall back to Date Added
            paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')
            monthly_activity = paper_df.groupby('Month').size().to_dict()
    else:
        # Use Date Added if Date Read not available
        paper_df['Month'] = paper_df['Date Added'].dt.strftime('%Y-%m')
        monthly_activity = paper_df.groupby('Month').size().to_dict()

    # Completion rate
    completion_rate = (read_papers / total_papers * 100) if total_papers > 0 else 0

    # Recent trends in topics should be based on recently READ papers, not added
    recent_topics = []
    if 'Date Read' in paper_df.columns:
        three_months_ago = datetime.now() - timedelta(days=90)
        recent_read_papers = paper_df[(paper_df['Reading Status'] == 'Read') & 
                                     (paper_df['Date Read'] >= three_months_ago)]
        if len(recent_read_papers) > 0:
            recent_topic_counts = Counter()
            for topics in recent_read_papers['Topics']:
                if isinstance(topics, list):
                    for topic in topics:
                        recent_topic_counts[topic] += 1
            recent_topics = [topic for topic, _ in recent_topic_counts.most_common(5)]
    
    # Fall back to Date Added if no recent read data
    if not recent_topics:
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
        "recent_topics": recent_topics,
        "author_engagement": author_engagement,
        "category_distribution": category_distribution,
        "version_distribution": version_distribution,
        "publication_timeline": publication_timeline,
        "arxiv_statistics": arxiv_statistics
    }


def calculate_reading_time_metrics(paper_df):
    """Calculate metrics based on reading time differences between dates."""
    if paper_df.empty:
        return {}
    
    # Ensure date columns are datetime
    date_columns = ['Date Added', 'Date Published', 'Date Read']
    for col in date_columns:
        if col in paper_df.columns:
            paper_df[col] = pd.to_datetime(paper_df[col], errors='coerce')
    
    metrics = {}
    
    # Time from adding to reading (processing time)
    read_papers = paper_df[(paper_df['Reading Status'] == 'Read') & 
                          paper_df['Date Read'].notna() & 
                          paper_df['Date Added'].notna()]
    
    if len(read_papers) > 0:
        processing_times = (read_papers['Date Read'] - read_papers['Date Added']).dt.days
        processing_times = processing_times[processing_times >= 0]  # Filter out negative values
        
        if len(processing_times) > 0:
            metrics['avg_processing_time_days'] = processing_times.mean()
            metrics['median_processing_time_days'] = processing_times.median()
            metrics['min_processing_time_days'] = processing_times.min()
            metrics['max_processing_time_days'] = processing_times.max()
            metrics['processing_time_std'] = processing_times.std()
    
    # Time from publication to reading (publication lag)
    papers_with_pub_date = paper_df[(paper_df['Reading Status'] == 'Read') & 
                                   paper_df['Date Read'].notna() & 
                                   paper_df['Date Published'].notna()]
    
    if len(papers_with_pub_date) > 0:
        publication_lags = (papers_with_pub_date['Date Read'] - papers_with_pub_date['Date Published']).dt.days
        publication_lags = publication_lags[publication_lags >= 0]  # Filter out future readings
        
        if len(publication_lags) > 0:
            metrics['avg_publication_lag_days'] = publication_lags.mean()
            metrics['median_publication_lag_days'] = publication_lags.median()
            metrics['reading_recency_score'] = 1 / (1 + publication_lags.mean() / 365)  # Score 0-1, higher = more recent
    
    # Reading velocity (papers per time period)
    if len(read_papers) > 1:
        reading_span = (read_papers['Date Read'].max() - read_papers['Date Read'].min()).days
        if reading_span > 0:
            metrics['reading_velocity_papers_per_month'] = len(read_papers) * 30 / reading_span
    
    # Reading consistency over time
    if len(read_papers) >= 3:
        # Group by month and count readings
        monthly_readings = read_papers.groupby(read_papers['Date Read'].dt.to_period('M')).size()
        if len(monthly_readings) > 1:
            cv = monthly_readings.std() / monthly_readings.mean() if monthly_readings.mean() > 0 else 0
            metrics['reading_consistency_score'] = max(0, 1 - cv)  # Higher score = more consistent
    
    # Current backlog age analysis
    unread_papers = paper_df[paper_df['Reading Status'].isin(['Want to Read', 'Reading'])]
    if len(unread_papers) > 0:
        current_time = pd.Timestamp.now()
        backlog_ages = (current_time - unread_papers['Date Added']).dt.days
        backlog_ages = backlog_ages[backlog_ages >= 0]
        
        if len(backlog_ages) > 0:
            metrics['avg_backlog_age_days'] = backlog_ages.mean()
            metrics['oldest_unread_days'] = backlog_ages.max()
            metrics['newest_unread_days'] = backlog_ages.min()
    
    # Reading pattern analysis
    if len(read_papers) > 0:
        # Day of week patterns
        dow_counts = read_papers['Date Read'].dt.day_name().value_counts()
        metrics['most_productive_day'] = dow_counts.index[0] if len(dow_counts) > 0 else None
        
        # Monthly reading trends
        monthly_counts = read_papers.groupby(read_papers['Date Read'].dt.to_period('M')).size()
        if len(monthly_counts) >= 2:
            recent_avg = monthly_counts.tail(3).mean()
            earlier_avg = monthly_counts.head(len(monthly_counts) - 3).mean() if len(monthly_counts) > 3 else recent_avg
            
            if recent_avg > earlier_avg * 1.2:
                metrics['reading_trend'] = 'increasing'
            elif recent_avg < earlier_avg * 0.8:
                metrics['reading_trend'] = 'decreasing'
            else:
                metrics['reading_trend'] = 'stable'
    
    return metrics


def calculate_advanced_research_metrics(paper_df):
    """Calculate advanced research metrics and patterns."""
    if paper_df.empty:
        return {}
    
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
    
    # Diversity metrics
    topic_counts = Counter()
    paper_topic_vectors = []
    all_topics = set()
    
    for topics in paper_df['Topics']:
        if isinstance(topics, list):
            for topic in topics:
                topic_counts[topic] += 1
                all_topics.add(topic)
    
    # Create topic vectors for each paper
    for topics in paper_df['Topics']:
        if isinstance(topics, list):
            vector = [1 if topic in topics else 0 for topic in sorted(all_topics)]
            paper_topic_vectors.append(vector)
        else:
            paper_topic_vectors.append([0] * len(all_topics))
    
    # Shannon diversity index for topics
    total_topic_instances = sum(topic_counts.values())
    shannon_diversity = 0
    if total_topic_instances > 0:
        for count in topic_counts.values():
            if count > 0:
                p = count / total_topic_instances
                shannon_diversity -= p * np.log(p)
    
    # Temporal analysis
    monthly_data = paper_df.groupby(paper_df['Date Added'].dt.to_period('M')).agg({
        'Title': 'count',
        'Reading Status': lambda x: sum(x == 'Read')
    }).rename(columns={'Title': 'added', 'Reading Status': 'completed'})
    
    # Reading consistency (coefficient of variation)
    reading_consistency = 0
    if len(monthly_data) > 1 and monthly_data['added'].std() > 0:
        reading_consistency = 1 - (monthly_data['added'].std() / monthly_data['added'].mean())
        reading_consistency = max(0, min(1, reading_consistency))
    
    # Research acceleration/deceleration
    research_momentum = "stable"
    if len(monthly_data) >= 3:
        recent_avg = monthly_data['added'].tail(3).mean()
        earlier_avg = monthly_data['added'].head(len(monthly_data) - 3).mean()
        
        if recent_avg > earlier_avg * 1.3:
            research_momentum = "accelerating"
        elif recent_avg < earlier_avg * 0.7:
            research_momentum = "decelerating"
    
    # Topic clustering coefficient
    topic_clustering = 0
    if len(paper_topic_vectors) > 1:
        try:
            # Calculate average cosine similarity between papers
            similarities = []
            for i in range(len(paper_topic_vectors)):
                for j in range(i + 1, len(paper_topic_vectors)):
                    v1, v2 = np.array(paper_topic_vectors[i]), np.array(paper_topic_vectors[j])
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        similarities.append(sim)
            
            if similarities:
                topic_clustering = np.mean(similarities)
        except Exception:
            topic_clustering = 0
    
    # Research domain expertise levels
    domain_expertise = {}
    for topic, count in topic_counts.items():
        if count >= 5:
            domain_expertise[topic] = "Expert"
        elif count >= 3:
            domain_expertise[topic] = "Proficient"
        elif count >= 2:
            domain_expertise[topic] = "Developing"
        else:
            domain_expertise[topic] = "Beginner"
    
    # Reading efficiency trends
    reading_efficiency_trend = []
    for period in monthly_data.index:
        period_data = monthly_data.loc[period]
        if period_data['added'] > 0:
            efficiency = period_data['completed'] / period_data['added']
            reading_efficiency_trend.append({
                'period': str(period),
                'efficiency': efficiency,
                'papers_added': period_data['added'],
                'papers_completed': period_data['completed']
            })
    
    # Research focus evolution
    focus_evolution = analyze_focus_evolution(paper_df, topic_counts)
    
    return {
        'shannon_diversity': round(shannon_diversity, 3),
        'reading_consistency': round(reading_consistency, 3),
        'research_momentum': research_momentum,
        'topic_clustering': round(topic_clustering, 3),
        'domain_expertise': domain_expertise,
        'reading_efficiency_trend': reading_efficiency_trend,
        'focus_evolution': focus_evolution,
        'total_unique_topics': len(all_topics),
        'avg_topics_per_paper': round(len([t for topics in paper_df['Topics'] for t in topics if isinstance(topics, list)]) / max(1, len(paper_df)), 2)
    }


def analyze_focus_evolution(paper_df, topic_counts):
    """Analyze how research focus has evolved over time."""
    if paper_df.empty:
        return {}
    
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
    paper_df = paper_df.sort_values('Date Added')
    
    # Split into time periods
    total_papers = len(paper_df)
    if total_papers < 6:
        return {"message": "Need more papers for focus evolution analysis"}
    
    third = total_papers // 3
    early_papers = paper_df.iloc[:third]
    middle_papers = paper_df.iloc[third:2*third]
    recent_papers = paper_df.iloc[2*third:]
    
    periods = {
        'early': early_papers,
        'middle': middle_papers,
        'recent': recent_papers
    }
    
    period_topics = {}
    for period_name, period_df in periods.items():
        topics = Counter()
        for topic_list in period_df['Topics']:
            if isinstance(topic_list, list):
                for topic in topic_list:
                    topics[topic] += 1
        period_topics[period_name] = dict(topics.most_common(5))
    
    # Calculate topic persistence
    all_period_topics = set()
    for topics in period_topics.values():
        all_period_topics.update(topics.keys())
    
    topic_persistence = {}
    for topic in all_period_topics:
        periods_present = sum(1 for period_topics_dict in period_topics.values() if topic in period_topics_dict)
        topic_persistence[topic] = periods_present / 3
    
    # Identify emerging and declining topics
    emerging_topics = []
    declining_topics = []
    
    for topic in all_period_topics:
        early_count = period_topics['early'].get(topic, 0)
        recent_count = period_topics['recent'].get(topic, 0)
        
        if recent_count > early_count * 1.5 and recent_count >= 2:
            emerging_topics.append(topic)
        elif early_count > recent_count * 1.5 and early_count >= 2:
            declining_topics.append(topic)
    
    return {
        'period_topics': period_topics,
        'topic_persistence': topic_persistence,
        'emerging_topics': emerging_topics,
        'declining_topics': declining_topics,
        'focus_stability': len([t for t, p in topic_persistence.items() if p >= 0.67]) / max(1, len(topic_persistence))
    }


def calculate_research_productivity_metrics(paper_df):
    """Calculate productivity and efficiency metrics."""
    if paper_df.empty:
        return {}
    
    # Ensure both date columns are properly formatted
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
    if 'Date Read' in paper_df.columns:
        paper_df['Date Read'] = pd.to_datetime(paper_df['Date Read'], errors='coerce')
    
    # Time-based metrics for acquisition
    date_range = paper_df['Date Added'].max() - paper_df['Date Added'].min()
    research_span_days = date_range.days if date_range.days > 0 else 1
    
    # Acquisition rate (papers added)
    papers_per_day = len(paper_df) / research_span_days
    papers_per_week = papers_per_day * 7
    papers_per_month = papers_per_day * 30
    
    # Reading productivity metrics (using Date Read for actual reading)
    read_papers = len(paper_df[paper_df['Reading Status'] == 'Read'])
    reading_papers = len(paper_df[paper_df['Reading Status'] == 'Reading'])
    
    # Calculate reading velocity based on Date Read
    reading_velocity_per_month = 0
    if 'Date Read' in paper_df.columns:
        read_papers_df = paper_df[(paper_df['Reading Status'] == 'Read') & paper_df['Date Read'].notna()]
        if len(read_papers_df) > 1:
            read_date_range = read_papers_df['Date Read'].max() - read_papers_df['Date Read'].min()
            read_span_days = read_date_range.days if read_date_range.days > 0 else 1
            reading_velocity_per_month = (len(read_papers_df) / read_span_days) * 30
    
    completion_rate = read_papers / len(paper_df) if len(paper_df) > 0 else 0
    active_reading_rate = reading_papers / len(paper_df) if len(paper_df) > 0 else 0
    
    # Backlog analysis
    backlog_size = len(paper_df[paper_df['Reading Status'] == 'Want to Read'])
    backlog_ratio = backlog_size / len(paper_df) if len(paper_df) > 0 else 0
    
    # Reading velocity over time
    monthly_completion = paper_df[paper_df['Reading Status'] == 'Read'].groupby(
        paper_df['Date Added'].dt.to_period('M')
    ).size()
    
    avg_monthly_completion = monthly_completion.mean() if len(monthly_completion) > 0 else 0
    
    # Time to completion estimation
    if avg_monthly_completion > 0:
        estimated_backlog_clearance = backlog_size / avg_monthly_completion
    else:
        estimated_backlog_clearance = float('inf')
    
    # Research intensity patterns
    daily_additions = paper_df.groupby(paper_df['Date Added'].dt.date).size()
    research_intensity = {
        'max_daily_additions': daily_additions.max() if len(daily_additions) > 0 else 0,
        'avg_daily_additions': daily_additions.mean() if len(daily_additions) > 0 else 0,
        'research_days_ratio': len(daily_additions) / research_span_days if research_span_days > 0 else 0
    }
    
    return {
        'research_span_days': research_span_days,
        'papers_per_day': round(papers_per_day, 4),
        'papers_per_week': round(papers_per_week, 2),
        'papers_per_month': round(papers_per_month, 1),
        'completion_rate': round(completion_rate, 3),
        'active_reading_rate': round(active_reading_rate, 3),
        'backlog_ratio': round(backlog_ratio, 3),
        'avg_monthly_completion': round(avg_monthly_completion, 1),
        'estimated_backlog_clearance_months': round(estimated_backlog_clearance, 1) if estimated_backlog_clearance != float('inf') else "∞",
        'research_intensity': research_intensity
    }


def analyze_topic_relationships(paper_df):
    """Analyze relationships and correlations between topics."""
    if paper_df.empty:
        return {}
    
    # Build co-occurrence matrix
    all_topics = set()
    for topics in paper_df['Topics']:
        if isinstance(topics, list):
            all_topics.update(topics)
    
    if len(all_topics) < 2:
        return {"message": "Need at least 2 different topics for relationship analysis"}
    
    topic_list = sorted(list(all_topics))
    n_topics = len(topic_list)
    cooccurrence_matrix = np.zeros((n_topics, n_topics))
    
    # Fill co-occurrence matrix
    for topics in paper_df['Topics']:
        if isinstance(topics, list) and len(topics) > 1:
            for i, topic1 in enumerate(topic_list):
                for j, topic2 in enumerate(topic_list):
                    if topic1 in topics and topic2 in topics and i != j:
                        cooccurrence_matrix[i][j] += 1
    
    # Calculate topic associations
    topic_associations = {}
    for i, topic1 in enumerate(topic_list):
        associations = []
        for j, topic2 in enumerate(topic_list):
            if i != j and cooccurrence_matrix[i][j] > 0:
                # Calculate association strength
                strength = cooccurrence_matrix[i][j]
                associations.append((topic2, strength))
        
        # Sort by strength and take top 3
        associations.sort(key=lambda x: x[1], reverse=True)
        topic_associations[topic1] = associations[:3]
    
    # Find topic communities/clusters
    topic_communities = find_topic_communities(cooccurrence_matrix, topic_list)
    
    # Calculate centrality measures
    topic_centrality = {}
    for i, topic in enumerate(topic_list):
        # Degree centrality (number of connections)
        degree = sum(1 for j in range(n_topics) if cooccurrence_matrix[i][j] > 0)
        
        # Strength centrality (sum of connection weights)
        strength = sum(cooccurrence_matrix[i])
        
        topic_centrality[topic] = {
            'degree': degree,
            'strength': strength,
            'normalized_degree': degree / (n_topics - 1) if n_topics > 1 else 0
        }
    
    return {
        'topic_associations': topic_associations,
        'topic_communities': topic_communities,
        'topic_centrality': topic_centrality,
        'cooccurrence_matrix': cooccurrence_matrix.tolist(),
        'topic_list': topic_list
    }


def find_topic_communities(cooccurrence_matrix, topic_list):
    """Find communities of related topics using simple clustering."""
    n_topics = len(topic_list)
    if n_topics < 3:
        return [topic_list]
    
    # Simple community detection based on strong connections
    communities = []
    visited = set()
    
    for i, topic in enumerate(topic_list):
        if topic in visited:
            continue
            
        community = [topic]
        visited.add(topic)
        
        # Find strongly connected topics
        for j, other_topic in enumerate(topic_list):
            if i != j and other_topic not in visited:
                if cooccurrence_matrix[i][j] >= 2:  # Threshold for strong connection
                    community.append(other_topic)
                    visited.add(other_topic)
        
        if len(community) > 1:
            communities.append(community)
        else:
            # Check if this topic connects to any existing community
            added_to_community = False
            for existing_community in communities:
                for community_topic in existing_community:
                    community_idx = topic_list.index(community_topic)
                    if cooccurrence_matrix[i][community_idx] >= 1:
                        existing_community.append(topic)
                        added_to_community = True
                        break
                if added_to_community:
                    break
            
            if not added_to_community:
                communities.append([topic])
    
    return communities


def generate_research_recommendations(paper_df, advanced_metrics, productivity_metrics):
    """Generate personalized research recommendations based on analysis."""
    if paper_df.empty:
        return []
    
    recommendations = []
    
    # Reading efficiency recommendations
    if productivity_metrics.get('completion_rate', 0) < 0.3:
        recommendations.append({
            'category': 'Reading Efficiency',
            'priority': 'High',
            'title': 'Improve Reading Completion Rate',
            'description': f"Your completion rate is {productivity_metrics.get('completion_rate', 0):.1%}. Focus on finishing papers before adding new ones.",
            'actionable_steps': [
                'Set a daily reading goal (e.g., 30 minutes)',
                'Use the Pomodoro technique for focused reading sessions',
                'Create reading notes to improve comprehension and retention'
            ]
        })
    
    # Topic diversity recommendations
    shannon_diversity = advanced_metrics.get('shannon_diversity', 0)
    if shannon_diversity < 1.5 and len(paper_df) > 10:
        recommendations.append({
            'category': 'Research Breadth',
            'priority': 'Medium',
            'title': 'Expand Research Horizons',
            'description': f"Your topic diversity score is {shannon_diversity:.2f}. Consider exploring adjacent research areas.",
            'actionable_steps': [
                'Search for papers that cite your current favorites',
                'Explore interdisciplinary journals',
                'Follow researchers who work across multiple domains'
            ]
        })
    elif shannon_diversity > 3.0:
        recommendations.append({
            'category': 'Research Focus',
            'priority': 'Medium',
            'title': 'Consider Specialization',
            'description': f"High topic diversity ({shannon_diversity:.2f}) suggests broad interests. Consider focusing on key areas for deeper expertise.",
            'actionable_steps': [
                'Identify your top 3-5 most important research areas',
                'Allocate 70% of reading time to core topics, 30% to exploration',
                'Create concept maps to connect different topics'
            ]
        })
    
    # Productivity recommendations
    papers_per_month = productivity_metrics.get('papers_per_month', 0)
    if papers_per_month > 20:
        recommendations.append({
            'category': 'Research Pace',
            'priority': 'High',
            'title': 'Quality Over Quantity',
            'description': f"Adding {papers_per_month:.1f} papers per month. Ensure quality engagement with each paper.",
            'actionable_steps': [
                'Implement a paper screening process',
                'Write summary notes for each paper',
                'Focus on highly cited or recent papers in your field'
            ]
        })
    elif papers_per_month < 2:
        recommendations.append({
            'category': 'Research Activity',
            'priority': 'Medium',
            'title': 'Increase Research Activity',
            'description': f"Only adding {papers_per_month:.1f} papers per month. Consider increasing research engagement.",
            'actionable_steps': [
                'Set up Google Scholar alerts for your topics',
                'Follow key conferences and journals in your field',
                'Schedule weekly research discovery sessions'
            ]
        })
    
    # Backlog management
    backlog_ratio = productivity_metrics.get('backlog_ratio', 0)
    if backlog_ratio > 0.6:
        recommendations.append({
            'category': 'Backlog Management',
            'priority': 'High',
            'title': 'Reduce Reading Backlog',
            'description': f"{backlog_ratio:.1%} of papers are unread. Implement backlog management strategies.",
            'actionable_steps': [
                'Review and remove papers that are no longer relevant',
                'Prioritize papers by importance and relevance',
                'Set a maximum backlog limit (e.g., 20 papers)'
            ]
        })
    
    # Topic relationship recommendations
    topic_communities = advanced_metrics.get('topic_communities', [])
    if len(topic_communities) > 3:
        recommendations.append({
            'category': 'Research Integration',
            'priority': 'Low',
            'title': 'Connect Research Areas',
            'description': f"You have {len(topic_communities)} distinct research communities. Look for connections between them.",
            'actionable_steps': [
                'Search for papers that bridge your different research areas',
                'Attend interdisciplinary conferences',
                'Consider collaboration opportunities across domains'
            ]
        })
    
    # Consistency recommendations
    reading_consistency = advanced_metrics.get('reading_consistency', 0)
    if reading_consistency < 0.5:
        recommendations.append({
            'category': 'Reading Habits',
            'priority': 'Medium',
            'title': 'Improve Reading Consistency',
            'description': f"Reading consistency score is {reading_consistency:.2f}. Develop more regular reading habits.",
            'actionable_steps': [
                'Schedule specific times for reading research papers',
                'Use a reading tracker or journal',
                'Set weekly reading goals rather than daily ones'
            ]
        })
    
    return sorted(recommendations, key=lambda x: {'High': 3, 'Medium': 2, 'Low': 1}[x['priority']], reverse=True)


def create_reading_time_visualizations(paper_df, reading_time_metrics):
    """Create visualizations for reading time analysis."""
    visualizations = {}
    
    if paper_df.empty:
        return visualizations
    
    # Ensure date columns are datetime
    date_columns = ['Date Added', 'Date Published', 'Date Read']
    for col in date_columns:
        if col in paper_df.columns:
            paper_df[col] = pd.to_datetime(paper_df[col], errors='coerce')
    
    # 1. Processing Time Distribution
    read_papers = paper_df[(paper_df['Reading Status'] == 'Read') & 
                          paper_df['Date Read'].notna() & 
                          paper_df['Date Added'].notna()]
    
    if len(read_papers) > 3:
        processing_times = (read_papers['Date Read'] - read_papers['Date Added']).dt.days
        processing_times = processing_times[processing_times >= 0]
        
        if len(processing_times) > 0:
            fig = px.histogram(
                x=processing_times,
                nbins=min(20, len(processing_times)),
                title="Distribution of Processing Times (Days from Adding to Reading)",
                labels={'x': 'Days', 'y': 'Count'}
            )
            fig.update_layout(showlegend=False)
            visualizations['processing_time_hist'] = fig
    
    # 2. Reading Timeline with Processing Times
    if len(read_papers) > 1:
        timeline_data = []
        for idx, row in read_papers.iterrows():
            if pd.notna(row['Date Added']) and pd.notna(row['Date Read']):
                processing_time = (row['Date Read'] - row['Date Added']).days
                timeline_data.append({
                    'Title': row['Title'][:50] + '...' if len(row['Title']) > 50 else row['Title'],
                    'Date Added': row['Date Added'],
                    'Date Read': row['Date Read'],
                    'Processing Time': processing_time,
                    'Status': 'Completed'
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            
            fig = go.Figure()
            
            # Add lines connecting date added to date read
            for _, row in timeline_df.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row['Date Added'], row['Date Read']],
                    y=[row['Title'], row['Title']],
                    mode='lines+markers',
                    line=dict(color='lightblue', width=2),
                    marker=dict(size=8, color=['red', 'green']),
                    name=f"{row['Processing Time']} days",
                    showlegend=False,
                    hovertemplate=f"<b>{row['Title']}</b><br>Processing: {row['Processing Time']} days<extra></extra>"
                ))
            
            fig.update_layout(
                title="Reading Timeline: From Addition to Completion",
                xaxis_title="Date",
                yaxis_title="Papers",
                height=max(400, len(timeline_df) * 30),
                hovermode='closest'
            )
            
            visualizations['reading_timeline'] = fig
    
    # 3. Publication Lag Analysis
    papers_with_pub = paper_df[(paper_df['Reading Status'] == 'Read') & 
                              paper_df['Date Read'].notna() & 
                              paper_df['Date Published'].notna()]
    
    if len(papers_with_pub) > 3:
        pub_lags = (papers_with_pub['Date Read'] - papers_with_pub['Date Published']).dt.days / 365  # Convert to years
        pub_lags = pub_lags[pub_lags >= 0]
        
        if len(pub_lags) > 0:
            # Create DataFrame for px.scatter
            scatter_data = pd.DataFrame({
                'Date Published': papers_with_pub['Date Published'].values,
                'Years Until Read': pub_lags.values,
                'Title': papers_with_pub['Title'].values
            })
            
            fig = px.scatter(
                scatter_data,
                x='Date Published',
                y='Years Until Read',
                hover_data=['Title'],
                title="Publication Lag: How Long After Publication Did You Read?",
                labels={'Date Published': 'Publication Date', 'Years Until Read': 'Years Until Read'}
            )
            
            # Add trend line
            if len(pub_lags) > 5:
                z = np.polyfit(papers_with_pub['Date Published'].astype(np.int64) // 10**9, pub_lags, 1)
                p = np.poly1d(z)
                fig.add_trace(go.Scatter(
                    x=papers_with_pub['Date Published'],
                    y=p(papers_with_pub['Date Published'].astype(np.int64) // 10**9),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                ))
            
            visualizations['publication_lag'] = fig
    
    # 4. Monthly Reading Velocity
    if len(read_papers) > 6:
        monthly_completion = read_papers.groupby(read_papers['Date Read'].dt.to_period('M')).size()
        
        fig = px.bar(
            x=[str(period) for period in monthly_completion.index],
            y=monthly_completion.values,
            title="Monthly Reading Completion Velocity",
            labels={'x': 'Month', 'y': 'Papers Completed'}
        )
        
        # Add average line
        avg_completion = monthly_completion.mean()
        fig.add_hline(y=avg_completion, line_dash="dash", line_color="red", 
                     annotation_text=f"Average: {avg_completion:.1f}")
        
        visualizations['monthly_velocity'] = fig
    
    # 5. Reading Efficiency Over Time
    if len(read_papers) > 5:
        # Calculate cumulative reading efficiency
        read_papers_sorted = read_papers.sort_values('Date Read')
        processing_times = (read_papers_sorted['Date Read'] - read_papers_sorted['Date Added']).dt.days
        
        # Rolling average of processing times
        window_size = min(5, len(processing_times) // 2)
        rolling_avg = processing_times.rolling(window=window_size, center=True).mean()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=read_papers_sorted['Date Read'],
            y=processing_times,
            mode='markers',
            name='Individual Papers',
            marker=dict(size=6, opacity=0.6)
        ))
        
        fig.add_trace(go.Scatter(
            x=read_papers_sorted['Date Read'],
            y=rolling_avg,
            mode='lines',
            name=f'Rolling Average ({window_size} papers)',
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Reading Efficiency Over Time",
            xaxis_title="Date Read",
            yaxis_title="Processing Time (Days)",
            hovermode='x unified'
        )
        
        visualizations['efficiency_trend'] = fig
    
    return visualizations


def create_advanced_visualizations(paper_df, advanced_metrics, topic_relationships):
    """Create advanced visualizations for research analysis."""
    visualizations = {}
    
    if paper_df.empty:
        return visualizations
    
    # 1. Research productivity heatmap
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
    paper_df['WeekDay'] = paper_df['Date Added'].dt.day_name()
    paper_df['Week'] = paper_df['Date Added'].dt.isocalendar().week
    
    # Create productivity heatmap data
    heatmap_data = paper_df.groupby(['WeekDay', 'Week']).size().reset_index(name='Papers')
    
    if not heatmap_data.empty:
        heatmap_fig = px.density_heatmap(
            heatmap_data, 
            x='Week', 
            y='WeekDay', 
            z='Papers',
            title="Research Activity Heatmap",
            color_continuous_scale='Viridis'
        )
        visualizations['productivity_heatmap'] = heatmap_fig
    
    # 2. Topic evolution sunburst
    focus_evolution = advanced_metrics.get('focus_evolution', {})
    if isinstance(focus_evolution, dict) and 'period_topics' in focus_evolution:
        sunburst_data = []
        for period, topics in focus_evolution['period_topics'].items():
            for topic, count in topics.items():
                sunburst_data.append({
                    'period': period.capitalize(),
                    'topic': topic,
                    'count': count,
                    'id': f"{period}_{topic}",
                    'parent': period.capitalize()
                })
        
        # Add period nodes
        for period in focus_evolution['period_topics'].keys():
            sunburst_data.append({
                'period': period.capitalize(),
                'topic': '',
                'count': sum(focus_evolution['period_topics'][period].values()),
                'id': period.capitalize(),
                'parent': ''
            })
        
        if sunburst_data:
            sunburst_df = pd.DataFrame(sunburst_data)
            sunburst_fig = px.sunburst(
                sunburst_df,
                path=['period', 'topic'] if 'topic' in sunburst_df.columns else ['period'],
                values='count',
                title="Research Focus Evolution"
            )
            visualizations['topic_evolution'] = sunburst_fig
    
    # 3. Topic relationship network (enhanced)
    if 'cooccurrence_matrix' in topic_relationships and topic_relationships['topic_list']:
        topic_list = topic_relationships['topic_list']
        matrix = np.array(topic_relationships['cooccurrence_matrix'])
        
        # Create network graph
        network_fig = go.Figure()
        
        # Calculate positions for nodes (circular layout)
        n_topics = len(topic_list)
        angles = [2 * np.pi * i / n_topics for i in range(n_topics)]
        node_x = [np.cos(angle) for angle in angles]
        node_y = [np.sin(angle) for angle in angles]
        
        # Add edges
        edge_x, edge_y = [], []
        edge_weights = []
        
        for i in range(n_topics):
            for j in range(i + 1, n_topics):
                if matrix[i][j] > 0:
                    edge_x.extend([node_x[i], node_x[j], None])
                    edge_y.extend([node_y[i], node_y[j], None])
                    edge_weights.append(matrix[i][j])
        
        # Add edges trace
        network_fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='rgba(125,125,125,0.5)'),
            hoverinfo='none'
        ))
        
        # Calculate node sizes based on centrality
        centrality = topic_relationships.get('topic_centrality', {})
        node_sizes = [centrality.get(topic, {}).get('strength', 1) * 5 + 10 for topic in topic_list]
        node_colors = [centrality.get(topic, {}).get('degree', 0) for topic in topic_list]
        
        # Add nodes trace
        network_fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Connections")
            ),
            text=topic_list,
            textposition="top center",
            hovertemplate='<b>%{text}</b><br>Connections: %{marker.color}<extra></extra>'
        ))
        
        network_fig.update_layout(
            title="Enhanced Topic Relationship Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Node size = relationship strength<br>Color = number of connections",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002, xanchor='left', yanchor='bottom',
                    font=dict(color="grey", size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        visualizations['enhanced_network'] = network_fig
    
    return visualizations


def create_comprehensive_research_dashboard(paper_df, analysis_data, advanced_metrics, productivity_metrics):
    """Create a comprehensive research dashboard with key insights."""
    if paper_df.empty:
        return None
    
    # Create a summary dashboard
    fig = go.Figure()
    
    # Add multiple subplots
    from plotly.subplots import make_subplots
    
    dashboard_fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Reading Progress Over Time', 'Topic Distribution', 
                       'Productivity Metrics', 'Research Focus Evolution'),
        specs=[[{"secondary_y": True}, {"type": "pie"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # 1. Reading progress over time
    paper_df['Date Added'] = pd.to_datetime(paper_df['Date Added'], errors='coerce')
    monthly_progress = paper_df.groupby([
        paper_df['Date Added'].dt.to_period('M'),
        'Reading Status'
    ]).size().unstack(fill_value=0)
    
    if not monthly_progress.empty:
        months = [str(m) for m in monthly_progress.index]
        
        for status in ['Want to Read', 'Reading', 'Read']:
            if status in monthly_progress.columns:
                dashboard_fig.add_trace(
                    go.Scatter(
                        x=months,
                        y=monthly_progress[status].values,
                        name=status,
                        mode='lines+markers'
                    ),
                    row=1, col=1
                )
    
    # 2. Topic distribution pie chart
    topic_dist = analysis_data.get('topic_distribution', {})
    if topic_dist:
        top_topics = dict(sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)[:8])
        dashboard_fig.add_trace(
            go.Pie(
                labels=list(top_topics.keys()),
                values=list(top_topics.values()),
                name="Topics"
            ),
            row=1, col=2
        )
    
    # 3. Productivity metrics bar chart
    prod_metrics = {
        'Completion Rate': productivity_metrics.get('completion_rate', 0) * 100,
        'Reading Consistency': advanced_metrics.get('reading_consistency', 0) * 100,
        'Topic Diversity': advanced_metrics.get('shannon_diversity', 0) * 20,  # Scale for visibility
        'Research Activity': min(productivity_metrics.get('papers_per_month', 0) * 10, 100)  # Cap at 100
    }
    
    dashboard_fig.add_trace(
        go.Bar(
            x=list(prod_metrics.keys()),
            y=list(prod_metrics.values()),
            name="Metrics (%)",
            marker_color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']
        ),
        row=2, col=1
    )
    
    # 4. Research focus evolution
    focus_evolution = advanced_metrics.get('focus_evolution', {})
    if isinstance(focus_evolution, dict) and 'period_topics' in focus_evolution:
        periods = list(focus_evolution['period_topics'].keys())
        topic_counts = []
        
        for period in periods:
            count = len(focus_evolution['period_topics'][period])
            topic_counts.append(count)
        
        dashboard_fig.add_trace(
            go.Scatter(
                x=periods,
                y=topic_counts,
                mode='lines+markers+text',
                text=topic_counts,
                textposition="top center",
                name="Active Topics",
                marker=dict(size=12, color='#E91E63')
            ),
            row=2, col=2
        )
    
    # Update layout
    dashboard_fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Research Analytics Dashboard",
        title_x=0.5
    )
    
    return dashboard_fig


def generate_research_insights_text(analysis_data, advanced_metrics, productivity_metrics):
    """Generate comprehensive text-based research insights."""
    insights = []
    
    total_papers = analysis_data.get('total_papers', 0)
    if total_papers == 0:
        return ["Add papers to generate insights about your research patterns."]
    
    # Research volume insights
    papers_per_month = productivity_metrics.get('papers_per_month', 0)
    if papers_per_month > 10:
        insights.append(
            f"🔥 **High Research Activity**: You're adding {papers_per_month:.1f} papers per month, which indicates "
            f"strong research engagement. Make sure you're maintaining quality over quantity."
        )
    elif papers_per_month < 1:
        insights.append(
            f"🌱 **Growing Research Base**: With {papers_per_month:.1f} papers per month, you're building your "
            f"research foundation steadily. Consider setting up alerts for your key topics."
        )
    
    # Reading efficiency insights
    completion_rate = productivity_metrics.get('completion_rate', 0)
    if completion_rate > 0.8:
        insights.append(
            f"⭐ **Excellent Reading Discipline**: {completion_rate:.1%} completion rate shows strong follow-through. "
            f"You effectively finish what you start reading."
        )
    elif completion_rate < 0.3:
        insights.append(
            f"📚 **Reading Backlog Alert**: {completion_rate:.1%} completion rate suggests you're adding papers faster "
            f"than reading them. Consider implementing a 'one-in, one-out' policy."
        )
    
    # Topic diversity insights
    shannon_diversity = advanced_metrics.get('shannon_diversity', 0)
    total_topics = advanced_metrics.get('total_unique_topics', 0)
    
    if shannon_diversity > 2.5:
        insights.append(
            f"🌈 **Broad Research Interests**: Shannon diversity of {shannon_diversity:.2f} across {total_topics} topics "
            f"indicates wide-ranging intellectual curiosity. Consider identifying 3-5 core areas for deeper focus."
        )
    elif shannon_diversity < 1.0 and total_topics > 3:
        insights.append(
            f"🎯 **Highly Focused Research**: Low diversity ({shannon_diversity:.2f}) suggests strong specialization. "
            f"This depth is valuable, but consider occasional exploration of adjacent fields."
        )
    
    # Research momentum insights
    momentum = advanced_metrics.get('research_momentum', 'stable')
    if momentum == 'accelerating':
        insights.append(
            f"🚀 **Research Momentum Building**: Your research activity is accelerating. This is great for "
            f"knowledge acquisition, but ensure you're giving adequate time for deep understanding."
        )
    elif momentum == 'decelerating':
        insights.append(
            f"⚡ **Momentum Opportunity**: Research activity is slowing down. Consider setting specific weekly "
            f"goals or exploring new topics to reignite your research passion."
        )
    
    # Topic clustering insights
    clustering = advanced_metrics.get('topic_clustering', 0)
    if clustering > 0.6:
        insights.append(
            f"🔗 **Well-Connected Research**: High topic clustering ({clustering:.2f}) indicates you're finding "
            f"papers that connect well thematically. This suggests good research coherence."
        )
    elif clustering < 0.2 and total_topics > 5:
        insights.append(
            f"🌐 **Diverse Research Landscape**: Low clustering ({clustering:.2f}) suggests you're exploring "
            f"independent research areas. Consider looking for bridges between your interests."
        )
    
    # Domain expertise insights
    domain_expertise = advanced_metrics.get('domain_expertise', {})
    expert_domains = [topic for topic, level in domain_expertise.items() if level == 'Expert']
    
    if expert_domains:
        insights.append(
            f"🏆 **Research Expertise Developed**: You've reached expert level in {', '.join(expert_domains[:3])}. "
            f"Consider sharing insights through blogs, papers, or teaching."
        )
    
    developing_domains = [topic for topic, level in domain_expertise.items() if level in ['Developing', 'Proficient']]
    if len(developing_domains) > 3:
        insights.append(
            f"🌟 **Skill Development Active**: You're developing expertise in {len(developing_domains)} areas. "
            f"Focus on 2-3 key domains to accelerate your path to expertise."
        )
    
    # Reading consistency insights
    consistency = advanced_metrics.get('reading_consistency', 0)
    if consistency > 0.8:
        insights.append(
            f"📈 **Consistent Reading Habits**: Excellent consistency score ({consistency:.2f}) indicates "
            f"regular research habits. This steady approach builds strong knowledge foundations."
        )
    elif consistency < 0.4:
        insights.append(
            f"📊 **Opportunity for Routine**: Reading consistency of {consistency:.2f} suggests irregular patterns. "
            f"Try scheduling specific research times or setting weekly goals."
        )
    
    # Research span insights
    research_span = productivity_metrics.get('research_span_days', 0)
    if research_span > 365:
        insights.append(
            f"📅 **Long-term Research Journey**: {research_span} days of research activity shows sustained "
            f"commitment to learning. Your knowledge compound interest is building!"
        )
    
    return insights[:8]  # Return top 8 insights


def calculate_research_quality_score(paper_df, analysis_data, advanced_metrics, productivity_metrics):
    """Calculate an overall research quality score."""
    if paper_df.empty:
        return 0, {}
    
    scores = {}
    weights = {}
    
    # Completion rate score (0-25 points)
    completion_rate = productivity_metrics.get('completion_rate', 0)
    scores['completion'] = min(completion_rate * 25, 25)
    weights['completion'] = 0.25
    
    # Reading consistency score (0-20 points)
    consistency = advanced_metrics.get('reading_consistency', 0)
    scores['consistency'] = consistency * 20
    weights['consistency'] = 0.20
    
    # Topic diversity score (0-20 points)
    diversity = advanced_metrics.get('shannon_diversity', 0)
    # Normalize diversity (optimal range 1.5-3.0)
    if diversity <= 1.5:
        diversity_score = (diversity / 1.5) * 15
    elif diversity <= 3.0:
        diversity_score = 15 + ((diversity - 1.5) / 1.5) * 5
    else:
        diversity_score = 20 - min((diversity - 3.0) * 2, 5)  # Penalty for too much diversity
    scores['diversity'] = diversity_score
    weights['diversity'] = 0.20
    
    # Research activity score (0-15 points)
    papers_per_month = productivity_metrics.get('papers_per_month', 0)
    # Optimal range 2-8 papers per month
    if papers_per_month <= 2:
        activity_score = (papers_per_month / 2) * 10
    elif papers_per_month <= 8:
        activity_score = 10 + ((papers_per_month - 2) / 6) * 5
    else:
        activity_score = 15 - min((papers_per_month - 8) * 0.5, 5)  # Penalty for too many
    scores['activity'] = activity_score
    weights['activity'] = 0.15
    
    # Topic clustering score (0-10 points)
    clustering = advanced_metrics.get('topic_clustering', 0)
    scores['clustering'] = clustering * 10
    weights['clustering'] = 0.10
    
    # Research momentum score (0-10 points)
    momentum = advanced_metrics.get('research_momentum', 'stable')
    momentum_scores = {'accelerating': 10, 'stable': 7, 'decelerating': 4}
    scores['momentum'] = momentum_scores.get(momentum, 7)
    weights['momentum'] = 0.10
    
    # Calculate weighted total
    total_score = sum(scores[key] * weights[key] / weights[key] * (weights[key] if key in weights else 1) 
                     for key in scores)
    
    # Normalize to 0-100
    total_score = min(total_score, 100)
    
    # Create detailed breakdown
    score_breakdown = {
        'total_score': round(total_score, 1),
        'grade': get_research_grade(total_score),
        'components': {
            'Reading Completion': round(scores['completion'], 1),
            'Reading Consistency': round(scores['consistency'], 1),
            'Topic Diversity': round(scores['diversity'], 1),
            'Research Activity': round(scores['activity'], 1),
            'Topic Integration': round(scores['clustering'], 1),
            'Research Momentum': round(scores['momentum'], 1)
        },
        'recommendations': get_score_recommendations(scores, total_score)
    }
    
    return total_score, score_breakdown


def get_research_grade(score):
    """Convert numeric score to letter grade."""
    if score >= 90:
        return "A+ (Research Master)"
    elif score >= 85:
        return "A (Expert Researcher)"
    elif score >= 80:
        return "A- (Advanced Researcher)"
    elif score >= 75:
        return "B+ (Proficient Researcher)"
    elif score >= 70:
        return "B (Good Researcher)"
    elif score >= 65:
        return "B- (Developing Researcher)"
    elif score >= 60:
        return "C+ (Active Learner)"
    elif score >= 55:
        return "C (Casual Reader)"
    elif score >= 50:
        return "C- (Occasional Reader)"
    else:
        return "D (Getting Started)"


def get_score_recommendations(scores, total_score):
    """Generate recommendations based on score components."""
    recommendations = []
    
    if scores['completion'] < 15:
        recommendations.append("Focus on completing more papers you start reading")
    
    if scores['consistency'] < 12:
        recommendations.append("Develop more regular reading habits")
    
    if scores['diversity'] < 12:
        recommendations.append("Consider exploring more diverse topics or focusing on fewer areas")
    
    if scores['activity'] < 8:
        recommendations.append("Increase your research paper discovery and reading frequency")
    
    if scores['clustering'] < 6:
        recommendations.append("Look for connections between your research topics")
    
    if scores['momentum'] < 6:
        recommendations.append("Set goals to maintain or increase research momentum")
    
    if total_score > 85:
        recommendations.append("Excellent research habits! Consider sharing your knowledge with others")
    
    return recommendations


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
    # Initialize session state first
    if 'paper' not in st.session_state:
        st.session_state.paper = load_paper()
    
    # Initialize success message state if not exists
    if 'success_message' not in st.session_state:
        st.session_state.success_message = None

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

    tab1, tab2, tab3, tab4 = st.tabs(["➕ Add Paper", "📋 View Collection", "📊 Analysis", "🔧 Auto-Fill"])

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
                    result = fetch_paper_details(arxiv_url)
                    if result is None or len(result) != 2:
                        st.error("Error: Unable to fetch paper details")
                    else:
                        paper_details, error = result
                        if error:
                            st.error(f"Error: {error}")
                        elif paper_details:
                            st.session_state['paper_details'] = paper_details
                            st.session_state['title'] = paper_details['title']
                            st.session_state['link'] = paper_details['link']
                            st.session_state['description'] = paper_details['description']
                            st.session_state.success_message = "📄 Paper details fetched successfully!"
                            st.rerun()
                        else:
                            st.error("Error: No paper details returned")
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

            col1, col2, col3 = st.columns(3)
            with col1:
                date_added = st.date_input(
                    "📅 Date Added",
                    value=datetime.strptime(
                        st.session_state.get('paper_details', {}).get('date_added', datetime.today().strftime('%Y-%m-%d')),
                        '%Y-%m-%d'
                    ).date()
                )
                reading_status = st.selectbox(
                    "📖 Reading Status",
                    ["Want to Read", "Reading", "Read"]
                )

            with col2:
                # Published date from arXiv or manual entry - defaults to Date Added
                pub_date_str = st.session_state.get('paper_details', {}).get('date_published', '')
                if pub_date_str:
                    try:
                        pub_date_default = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
                    except:
                        pub_date_default = date_added  # Default to Date Added
                else:
                    pub_date_default = date_added  # Default to Date Added
                
                date_published = st.date_input(
                    "📅 Published Date",
                    value=pub_date_default,
                    help="Publication date (defaults to Date Added)"
                )
                
                link = st.text_input("🌐 Link",
                                    value=st.session_state.get('paper_details', {}).get('link', ''),
                                    help="ArXiv or DOI link")

            with col3:
                # Date read - automatically set when status is "Read"
                show_date_read = reading_status == "Read"
                if show_date_read:
                    date_read = st.date_input(
                        "📅 Date Finished Reading",
                        value=datetime.today().date(),
                        help="When did you finish reading this paper?"
                    )
                else:
                    date_read = None
                    st.write("")  # Empty space for alignment
                    st.info("📅 Date read will be automatically set when status changes to 'Read'")
                
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
                    # Format dates using the new function
                    date_added_str = format_date_for_storage(date_added)
                    date_published_str = format_date_for_storage(date_published) if date_published else ''
                    date_read_str = format_date_for_storage(date_read) if date_read else ''
                    
                    if add_paper(title, reading_status, date_added_str, link, topic_list, 
                               description, date_published_str, date_read_str):
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
            # Row 1: Basic filters
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

            # Row 2: ArXiv-specific filters
            st.markdown("##### 🔬 ArXiv Filters")
            arxiv_col1, arxiv_col2, arxiv_col3 = st.columns([1, 1, 1])
            
            with arxiv_col1:
                # Authors filter
                all_authors = set()
                for authors in st.session_state.paper.get('Authors', []):
                    if isinstance(authors, list):
                        all_authors.update(authors)
                    elif isinstance(authors, str) and authors.strip():
                        all_authors.update([author.strip() for author in authors.split(',')])
                
                if all_authors:
                    author_filter = st.multiselect("👥 Filter by Authors", 
                                                   options=["All"] + sorted(list(all_authors)))
                else:
                    author_filter = []

            with arxiv_col2:
                # Primary Category filter
                all_primary_cats = set()
                for cat in st.session_state.paper.get('Primary Category', []):
                    if isinstance(cat, str) and cat.strip():
                        all_primary_cats.add(cat.strip())
                
                if all_primary_cats:
                    primary_cat_filter = st.multiselect("📊 Filter by Primary Category",
                                                        options=["All"] + sorted(list(all_primary_cats)))
                else:
                    primary_cat_filter = []

            with arxiv_col3:
                # Version filter
                all_versions = set()
                for version in st.session_state.paper.get('Version', []):
                    if isinstance(version, str) and version.strip():
                        all_versions.add(version.strip())
                
                if all_versions:
                    version_filter = st.multiselect("🔄 Filter by Version",
                                                    options=["All"] + sorted(list(all_versions)))
                else:
                    version_filter = []

            # Row 3: Additional filters
            additional_col1, additional_col2, additional_col3 = st.columns([1, 1, 1])
            
            with additional_col1:
                # Date range filters
                date_filter_type = st.selectbox("📅 Date Filter Type", 
                                               ["None", "Date Added", "Date Published", "Date Updated"])
                
            with additional_col2:
                # PDF availability filter
                pdf_filter = st.selectbox("📄 PDF Availability", 
                                         ["All", "Has PDF Link", "No PDF Link"])
                
            with additional_col3:
                # All Categories filter (for papers with multiple categories)
                all_categories = set()
                for categories in st.session_state.paper.get('All Categories', []):
                    if isinstance(categories, list):
                        all_categories.update(categories)
                    elif isinstance(categories, str) and categories.strip():
                        all_categories.update([cat.strip() for cat in categories.split(';')])
                
                if all_categories:
                    all_cat_filter = st.multiselect("📚 Filter by All Categories",
                                                    options=["All"] + sorted(list(all_categories)))
                else:
                    all_cat_filter = []

            # Date range selector (only shown if date filter is selected)
            date_range = None
            if date_filter_type != "None":
                date_range = st.date_input(f"Select {date_filter_type} Range",
                                         value=None,
                                         help=f"Filter papers by {date_filter_type.lower()}")
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    pass  # Valid range
                else:
                    date_range = None

        # Sorting options
        sort_col1, sort_col2 = st.columns([1, 1])
        with sort_col1:
            sort_by = st.selectbox("🔄 Sort by", 
                                 ["Date Added", "Date Read", "Date Published", "Date Updated", 
                                  "Title", "Reading Status", "Topics Count", "Primary Category", "Version"])
            if sort_by == "Date Read":
                st.caption("📖 Sorts by reading completion date (papers without read dates appear last)")
            elif sort_by == "Date Published":
                st.caption("📅 Sorts by original publication date on ArXiv")
            elif sort_by == "Date Updated":
                st.caption("🔄 Sorts by last update date on ArXiv")
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

        # Apply author filter
        if author_filter and "All" not in author_filter:
            def author_match(authors_data):
                if isinstance(authors_data, list):
                    return any(author in author_filter for author in authors_data)
                elif isinstance(authors_data, str) and authors_data.strip():
                    author_list = [author.strip() for author in authors_data.split(',')]
                    return any(author in author_filter for author in author_list)
                return False
            
            papers_to_display = papers_to_display[
                papers_to_display.get('Authors', pd.Series(dtype='object')).apply(author_match)]

        # Apply primary category filter
        if primary_cat_filter and "All" not in primary_cat_filter:
            papers_to_display = papers_to_display[
                papers_to_display.get('Primary Category', pd.Series(dtype='object')).isin(primary_cat_filter)]

        # Apply version filter
        if version_filter and "All" not in version_filter:
            papers_to_display = papers_to_display[
                papers_to_display.get('Version', pd.Series(dtype='object')).isin(version_filter)]

        # Apply all categories filter
        if all_cat_filter and "All" not in all_cat_filter:
            def all_categories_match(categories_data):
                if isinstance(categories_data, list):
                    return any(cat in all_cat_filter for cat in categories_data)
                elif isinstance(categories_data, str) and categories_data.strip():
                    cat_list = [cat.strip() for cat in categories_data.split(';')]
                    return any(cat in all_cat_filter for cat in cat_list)
                return False
            
            papers_to_display = papers_to_display[
                papers_to_display.get('All Categories', pd.Series(dtype='object')).apply(all_categories_match)]

        # Apply PDF availability filter
        if pdf_filter != "All":
            if pdf_filter == "Has PDF Link":
                papers_to_display = papers_to_display[
                    papers_to_display.get('PDF Link', pd.Series(dtype='object')).notna() & 
                    (papers_to_display.get('PDF Link', pd.Series(dtype='object')) != '')]
            elif pdf_filter == "No PDF Link":
                papers_to_display = papers_to_display[
                    papers_to_display.get('PDF Link', pd.Series(dtype='object')).isna() | 
                    (papers_to_display.get('PDF Link', pd.Series(dtype='object')) == '')]

        # Apply date range filter based on selected date type
        if date_range and len(date_range) == 2 and date_filter_type != "None":
            date_column = date_filter_type
            if date_column in papers_to_display.columns:
                papers_to_display[date_column] = pd.to_datetime(papers_to_display[date_column], errors='coerce')
                start_date, end_date = date_range
                papers_to_display = papers_to_display[
                    (papers_to_display[date_column].dt.date >= start_date) &
                    (papers_to_display[date_column].dt.date <= end_date)]

        # Apply text filter (enhanced to include new fields)
        if text_filter:
            text_filter_conditions = (
                papers_to_display['Title'].str.contains(text_filter, case=False, na=False) |
                papers_to_display['Description'].str.contains(text_filter, case=False, na=False) |
                papers_to_display['Topics'].apply(
                    lambda x: any(text_filter.lower() in t.lower() for t in x if isinstance(x, list)))
            )
            
            # Add author search
            if 'Authors' in papers_to_display.columns:
                def author_text_search(authors_data):
                    if isinstance(authors_data, list):
                        return any(text_filter.lower() in author.lower() for author in authors_data)
                    elif isinstance(authors_data, str):
                        return text_filter.lower() in authors_data.lower()
                    return False
                text_filter_conditions |= papers_to_display['Authors'].apply(author_text_search)
            
            # Add category searches
            for col in ['Primary Category', 'All Categories']:
                if col in papers_to_display.columns:
                    def category_text_search(cat_data):
                        if isinstance(cat_data, list):
                            return any(text_filter.lower() in cat.lower() for cat in cat_data)
                        elif isinstance(cat_data, str):
                            return text_filter.lower() in cat_data.lower()
                        return False
                    text_filter_conditions |= papers_to_display[col].apply(category_text_search)
            
            # Add comment search
            if 'Comment' in papers_to_display.columns:
                text_filter_conditions |= papers_to_display['Comment'].str.contains(text_filter, case=False, na=False)
                
            papers_to_display = papers_to_display[text_filter_conditions]

        # Calculate topics count for sorting
        papers_to_display['Topics Count'] = papers_to_display['Topics'].apply(
            lambda x: len(x) if isinstance(x, list) else 0)

        # Apply sorting
        ascending = sort_direction == "Ascending"
        if sort_by == "Date Added":
            papers_to_display['Date Added'] = pd.to_datetime(papers_to_display['Date Added'])
            papers_to_display = papers_to_display.sort_values('Date Added', ascending=ascending)
        elif sort_by == "Date Read":
            # Handle Date Read sorting with proper null handling
            if 'Date Read' in papers_to_display.columns:
                papers_to_display['Date Read'] = pd.to_datetime(papers_to_display['Date Read'], errors='coerce')
                # Sort with nulls last (papers without read dates go to the end)
                papers_to_display = papers_to_display.sort_values('Date Read', ascending=ascending, na_position='last')
            else:
                st.warning("Date Read column not available. Please ensure papers have reading completion dates.")
                papers_to_display = papers_to_display.sort_values('Date Added', ascending=ascending)
        elif sort_by == "Date Published":
            # Handle Date Published sorting
            if 'Date Published' in papers_to_display.columns:
                papers_to_display['Date Published'] = pd.to_datetime(papers_to_display['Date Published'], errors='coerce')
                papers_to_display = papers_to_display.sort_values('Date Published', ascending=ascending, na_position='last')
            else:
                st.warning("Date Published column not available.")
                papers_to_display = papers_to_display.sort_values('Date Added', ascending=ascending)
        elif sort_by == "Date Updated":
            # Handle Date Updated sorting
            if 'Date Updated' in papers_to_display.columns:
                papers_to_display['Date Updated'] = pd.to_datetime(papers_to_display['Date Updated'], errors='coerce')
                papers_to_display = papers_to_display.sort_values('Date Updated', ascending=ascending, na_position='last')
            else:
                st.warning("Date Updated column not available.")
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
        elif sort_by == "Primary Category":
            if 'Primary Category' in papers_to_display.columns:
                papers_to_display = papers_to_display.sort_values('Primary Category', ascending=ascending, na_position='last')
            else:
                st.warning("Primary Category column not available.")
                papers_to_display = papers_to_display.sort_values('Title', ascending=ascending)
        elif sort_by == "Version":
            if 'Version' in papers_to_display.columns:
                # Sort versions numerically (v1, v2, v3, etc.)
                def version_sort_key(version):
                    if pd.isna(version) or not isinstance(version, str):
                        return 0
                    # Extract number from version string (e.g., 'v2' -> 2)
                    import re
                    match = re.search(r'v(\d+)', version)
                    return int(match.group(1)) if match else 0
                
                papers_to_display['Version Sort Key'] = papers_to_display['Version'].apply(version_sort_key)
                papers_to_display = papers_to_display.sort_values('Version Sort Key', ascending=ascending, na_position='last')
            else:
                st.warning("Version column not available.")
                papers_to_display = papers_to_display.sort_values('Title', ascending=ascending)

        # Paper edit form
        if hasattr(st.session_state, 'edit_mode') and st.session_state.edit_mode and hasattr(st.session_state,
                                                                                             'selected_paper_index'):
            st.markdown("### ✏️ Edit Paper")
            index = st.session_state.selected_paper_index
            paper = st.session_state.paper.loc[index]

            with st.form(key=f"edit_form_{index}"):
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    new_title = st.text_input("Title", value=paper['Title'])
                    new_status = st.selectbox(
                        "📖 Reading Status",
                        ["Want to Read", "Reading", "Read"],
                        index=["Want to Read", "Reading", "Read"].index(paper['Reading Status'])
                    )
                    new_date = st.date_input(
                        "📅 Date Added",
                        value=safe_date_parse(paper['Date Added'])
                    )

                with col2:
                    # Published date - defaults to Date Added
                    current_pub_date = safe_date_parse(paper.get('Date Published', '')) if 'Date Published' in paper else None
                    if not current_pub_date:
                        current_pub_date = safe_date_parse(paper['Date Added'])  # Default to Date Added
                    
                    new_date_published = st.date_input(
                        "📅 Published Date",
                        value=current_pub_date,
                        help="Publication date (defaults to Date Added)"
                    )
                    
                    new_link = st.text_input("🌐 Link", value=paper['Link'])

                with col3:
                    # Date read - automatically set when status is "Read"
                    show_date_read = new_status == "Read"
                    if show_date_read:
                        current_read_date = safe_date_parse(paper.get('Date Read', '')) if 'Date Read' in paper else None
                        new_date_read = st.date_input(
                            "📖 Date Finished Reading",
                            value=current_read_date if current_read_date else datetime.today().date(),
                            help="When did you finish reading this paper?"
                        )
                    else:
                        new_date_read = None
                        st.write("")  # Empty space for alignment
                        st.info("📅 Date read will be automatically set when status changes to 'Read'")
                    
                    new_topics = st.text_input("🏷️ Topics", value=", ".join(paper['Topics']) if isinstance(paper['Topics'], list) else "")

                new_description = st.text_area("📝 Notes", value=paper['Description'], height=150)

                col1, col2 = st.columns([1, 1])
                with col1:
                    submitted = st.form_submit_button("💾 Save Changes")
                with col2:
                    cancel = st.form_submit_button("❌ Cancel")

                if submitted:
                    # Format dates using the new function
                    formatted_date_added = format_date_for_storage(new_date)
                    formatted_date_published = format_date_for_storage(new_date_published)
                    formatted_date_read = format_date_for_storage(new_date_read)
                    
                    # Update the paper with all fields including new date fields
                    edit_paper(index, new_title, new_status, formatted_date_added, 
                             new_link, ", ".join([t.strip() for t in new_topics.split(",") if t.strip()]), 
                             new_description,
                             formatted_date_published,
                             formatted_date_read)
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
                    date_str = ""
                    if not pd.isna(paper['Date Added']):
                        try:
                            date_added = pd.to_datetime(paper['Date Added'])
                            if not pd.isna(date_added):
                                date_str = date_added.strftime('%Y-%m-%d')
                        except (ValueError, AttributeError):
                            pass
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

                            # Display publication and reading dates
                            date_info_parts = []
                            if 'Date Published' in paper and not pd.isna(paper['Date Published']):
                                try:
                                    pub_datetime = pd.to_datetime(paper['Date Published'])
                                    if not pd.isna(pub_datetime):
                                        pub_date = pub_datetime.strftime('%Y-%m-%d')
                                        date_info_parts.append(f"📅 Published: {pub_date}")
                                except (ValueError, AttributeError):
                                    pass
                            
                            if 'Date Read' in paper and not pd.isna(paper['Date Read']):
                                try:
                                    read_datetime = pd.to_datetime(paper['Date Read'])
                                    if not pd.isna(read_datetime):
                                        read_date = read_datetime.strftime('%Y-%m-%d')
                                        date_info_parts.append(f"📖 Completed: {read_date}")
                                except (ValueError, AttributeError):
                                    pass
                            
                            if date_info_parts:
                                st.markdown(" | ".join(date_info_parts))

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
        
        # Create individual metric cards for better visibility
        metric_tabs = st.tabs(["📚 Research Corpus", "✅ Completion Rate", "🚀 Research Velocity", "⏱️ Time to Insights"])
        
        with metric_tabs[0]:
            st.metric(
                label="Research Corpus",
                value=f"{analysis_data['total_papers']:,}",
                help="Total number of papers in your research collection"
            )
            st.progress(min(analysis_data["total_papers"] / 100, 1.0))
            if analysis_data["total_papers"] < 10:
                st.info("🎯 **Goal**: Build a collection of 10+ papers for meaningful insights")
            elif analysis_data["total_papers"] < 50:
                st.success("✅ **Good progress**: You have a solid foundation for research analysis")
            else:
                st.success("🏆 **Excellent**: You have an extensive research collection")
        
        with metric_tabs[1]:
            completion_rate = analysis_data.get('completion_rate', 0)
            st.metric(
                label="Completion Rate",
                value=f"{completion_rate:.1f}%",
                delta=f"{completion_rate - 50:.1f}% vs 50% target",
                help="Percentage of papers you have completed reading"
            )
            st.progress(completion_rate / 100)
            if completion_rate < 30:
                st.warning("⚠️ **Focus needed**: Consider completing more papers before adding new ones")
            elif completion_rate < 70:
                st.info("📈 **Good balance**: You're making steady progress through your collection")
            else:
                st.success("🌟 **Excellent**: High completion rate indicates efficient reading habits")
        
        with metric_tabs[2]:
            velocity = analysis_data.get('reading_velocity', 0)
            st.metric(
                label="Research Velocity",
                value=f"{velocity:.2f} papers/month",
                help="Average number of papers you read or add per month"
            )
            velocity_progress = min(velocity / 10, 1.0)  # Scale to 10 papers/month max
            st.progress(velocity_progress)
            if velocity < 2:
                st.info("🐢 **Steady pace**: Quality over quantity approach")
            elif velocity < 5:
                st.success("⚡ **Good momentum**: Balanced reading pace")
            else:
                st.success("🚀 **High velocity**: Rapid knowledge acquisition")
        
        with metric_tabs[3]:
            if analysis_data["read_papers"] > 0:
                time_to_insights = (analysis_data['avg_reading_time'] * analysis_data['total_papers']) / analysis_data['read_papers']
                st.metric(
                    label="Time to Insights",
                    value=f"{time_to_insights:.1f} days",
                    help="Estimated time to gain insights from your research corpus"
                )
                st.progress(min(30 / time_to_insights, 1.0))  # 30 days is reasonable target
                if time_to_insights > 60:
                    st.warning("⏳ **Long timeline**: Consider focusing on key papers first")
                else:
                    st.success("💡 **Good timeline**: Reasonable time to complete current corpus")
            else:
                st.metric(
                    label="Time to Insights",
                    value="N/A",
                    help="Complete reading some papers to calculate this metric"
                )
                st.info("📖 Start reading papers to track your time to insights")

        # Research progress tracking - One visualization at a time for better detail
        st.subheader("📚 Research Progress Analysis")
        
        # Create tabs for different progress views
        progress_tabs = st.tabs(["📊 Reading Status", "🎯 Research Impact Metrics", "📈 Research Momentum", "🔍 Research Profile"])
        
        with progress_tabs[0]:
            st.markdown("#### Papers by Reading Status")
            status_labels = ["Read", "Reading", "Want to Read"]
            status_values = [analysis_data["read_papers"], analysis_data["reading_papers"], analysis_data["want_to_read_papers"]]
            
            # Create an enhanced pie chart with better interactivity
            status_fig = px.pie(
                names=status_labels, 
                values=status_values,
                color=status_labels,
                color_discrete_map={'Read': '#4CAF50', 'Reading': '#FFC107', 'Want to Read': '#2196F3'},
                hole=0.4,
                title="Reading Status Distribution"
            )
            status_fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Papers: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            status_fig.update_layout(
                height=500,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(status_fig, use_container_width=True)
            
            # Add detailed breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📖 Read", f"{analysis_data['read_papers']:,}", help="Papers you have completed")
            with col2:
                st.metric("📚 Reading", f"{analysis_data['reading_papers']:,}", help="Papers you are currently reading")
            with col3:
                st.metric("📋 Want to Read", f"{analysis_data['want_to_read_papers']:,}", help="Papers in your reading queue")
        
        with progress_tabs[1]:
            st.markdown("#### Research Impact Metrics")
            if analysis_data["total_papers"] > 0:
                # Calculate impact metrics using the provided function
                impact_metrics = calculate_research_impact_metrics(st.session_state.paper)
                
                # Display each metric with detailed explanation
                metrics_data = [
                    {
                        'name': 'Knowledge Breadth',
                        'value': impact_metrics['knowledge_breadth'],
                        'description': 'Number of unique topics per paper - higher values indicate broader research interests',
                        'optimal_range': '0.5 - 1.5',
                        'emoji': '🌐'
                    },
                    {
                        'name': 'Knowledge Depth', 
                        'value': impact_metrics['knowledge_depth'],
                        'description': 'Average papers per topic - higher values indicate deeper specialization',
                        'optimal_range': '2.0 - 5.0',
                        'emoji': '🔬'
                    },
                    {
                        'name': 'Research Efficiency',
                        'value': impact_metrics['research_efficiency'],
                        'description': 'Ratio of completed papers to total papers - measures reading completion',
                        'optimal_range': '0.6 - 0.9',
                        'emoji': '⚡'
                    },
                    {
                        'name': 'Topic Concentration',
                        'value': impact_metrics['topic_concentration'],
                        'description': 'How focused your research is on main topics vs diversified',
                        'optimal_range': '0.3 - 0.7',
                        'emoji': '🎯'
                    },
                    {
                        'name': 'Reading Consistency',
                        'value': impact_metrics['reading_consistency'],
                        'description': 'Regularity of your reading habits over time',
                        'optimal_range': '0.5 - 1.0',
                        'emoji': '📊'
                    },
                    {
                        'name': 'Exploration Ratio',
                        'value': impact_metrics['exploration_vs_exploitation'],
                        'description': 'Balance between exploring new topics vs deepening existing ones',
                        'optimal_range': '0.3 - 0.7',
                        'emoji': '🗺️'
                    },
                    # NEW ARXIV METRICS
                    {
                        'name': 'ArXiv Engagement',
                        'value': impact_metrics['arxiv_engagement'],
                        'description': 'Percentage of papers from ArXiv - indicates engagement with cutting-edge research',
                        'optimal_range': '0.3 - 0.8',
                        'emoji': '🔬'
                    },
                    {
                        'name': 'Category Diversity',
                        'value': impact_metrics['category_diversity'],
                        'description': 'Variety of ArXiv categories in your research - higher values show interdisciplinary interests',
                        'optimal_range': '0.2 - 0.6',
                        'emoji': '🎨'
                    },
                    {
                        'name': 'Version Awareness',
                        'value': impact_metrics['version_awareness'],
                        'description': 'How often you read updated versions of papers - shows attention to evolving research',
                        'optimal_range': '0.1 - 0.4',
                        'emoji': '🔄'
                    },
                    {
                        'name': 'Publication Recency',
                        'value': impact_metrics['publication_recency'],
                        'description': 'How recent the papers you read are - higher values indicate focus on latest research',
                        'optimal_range': '0.4 - 0.8',
                        'emoji': '📅'
                    }
                ]
                
                # Create individual metric displays
                for i, metric in enumerate(metrics_data):
                    with st.expander(f"{metric['emoji']} {metric['name']}: {metric['value']:.3f}", expanded=(i < 2)):
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric(metric['name'], f"{metric['value']:.3f}")
                            st.caption(f"Optimal: {metric['optimal_range']}")
                        with col2:
                            st.write(metric['description'])
                            # Create a simple gauge-like progress bar
                            progress_val = min(metric['value'], 1.0) if metric['value'] <= 1.0 else min(metric['value'] / 5.0, 1.0)
                            st.progress(progress_val)
            else:
                st.info("Add papers to see research impact metrics")
                
        with progress_tabs[2]:
            st.markdown("#### Research Momentum Analysis")
            if analysis_data["monthly_activity"]:
                # Check if we have the new structure with separate reading and acquisition activity
                if isinstance(analysis_data["monthly_activity"], dict) and "reading_activity" in analysis_data["monthly_activity"]:
                    # Enhanced momentum analysis with separate metrics
                    reading_activity = analysis_data["monthly_activity"]["reading_activity"]
                    acquisition_activity = analysis_data["monthly_activity"]["acquisition_activity"]
                    
                    # Create tabs for different activity views
                    activity_tabs = st.tabs(["📖 Reading Activity", "📥 Paper Acquisition", "📊 Combined View"])
                    
                    with activity_tabs[0]:
                        st.markdown("##### Papers Read Over Time (Based on Date Read)")
                        if reading_activity:
                            # Create interactive line chart for reading activity
                            months = list(reading_activity.keys())
                            values = list(reading_activity.values())
                            
                            reading_fig = px.line(
                                x=months, 
                                y=values,
                                title="Reading Activity Over Time",
                                labels={'x': 'Month', 'y': 'Papers Read'},
                                markers=True,
                                color_discrete_sequence=['#28a745']  # Green for reading
                            )
                            reading_fig.update_traces(
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>Papers Read: %{y}<extra></extra>'
                            )
                            reading_fig.update_layout(height=400)
                            st.plotly_chart(reading_fig, use_container_width=True)
                            
                            # Calculate reading momentum metrics
                            if len(reading_activity) >= 3:
                                recent_months = list(reading_activity.items())[-3:]
                                earlier_months = list(reading_activity.items())[:-3] if len(reading_activity) > 3 else []
                                
                                recent_avg = sum(month[1] for month in recent_months) / len(recent_months)
                                if earlier_months:
                                    earlier_avg = sum(month[1] for month in earlier_months) / len(earlier_months)
                                    reading_momentum_change = ((recent_avg - earlier_avg) / max(earlier_avg, 0.1)) * 100
                                else:
                                    reading_momentum_change = 0
                                    
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Recent Reading Rate", f"{recent_avg:.1f} papers/month")
                                with col2:
                                    st.metric("Reading Momentum", f"{reading_momentum_change:+.1f}%", 
                                            delta=f"{reading_momentum_change:.1f}%")
                                with col3:
                                    trend = "📈 Accelerating" if reading_momentum_change > 20 else "📉 Decelerating" if reading_momentum_change < -20 else "➡️ Stable"
                                    st.metric("Reading Trend", trend)
                                    
                                st.info("📖 **Reading Momentum**: Based on when you actually completed papers")
                        else:
                            st.info("No completed papers with reading dates yet. Mark papers as 'Read' and add completion dates to see reading momentum.")
                    
                    with activity_tabs[1]:
                        st.markdown("##### Paper Acquisition Over Time (Based on Date Added)")
                        if acquisition_activity:
                            # Create interactive line chart for acquisition activity
                            months = list(acquisition_activity.keys())
                            values = list(acquisition_activity.values())
                            
                            acquisition_fig = px.line(
                                x=months, 
                                y=values,
                                title="Paper Acquisition Over Time",
                                labels={'x': 'Month', 'y': 'Papers Added'},
                                markers=True,
                                color_discrete_sequence=['#007bff']  # Blue for acquisition
                            )
                            acquisition_fig.update_traces(
                                line=dict(width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>Papers Added: %{y}<extra></extra>'
                            )
                            acquisition_fig.update_layout(height=400)
                            st.plotly_chart(acquisition_fig, use_container_width=True)
                            
                            # Calculate acquisition momentum metrics
                            if len(acquisition_activity) >= 3:
                                recent_months = list(acquisition_activity.items())[-3:]
                                earlier_months = list(acquisition_activity.items())[:-3] if len(acquisition_activity) > 3 else []
                                
                                recent_avg = sum(month[1] for month in recent_months) / len(recent_months)
                                if earlier_months:
                                    earlier_avg = sum(month[1] for month in earlier_months) / len(earlier_months)
                                    acquisition_momentum_change = ((recent_avg - earlier_avg) / max(earlier_avg, 0.1)) * 100
                                else:
                                    acquisition_momentum_change = 0
                                    
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Recent Acquisition Rate", f"{recent_avg:.1f} papers/month")
                                with col2:
                                    st.metric("Acquisition Momentum", f"{acquisition_momentum_change:+.1f}%", 
                                            delta=f"{acquisition_momentum_change:.1f}%")
                                with col3:
                                    trend = "📈 Accelerating" if acquisition_momentum_change > 20 else "📉 Decelerating" if acquisition_momentum_change < -20 else "➡️ Stable"
                                    st.metric("Acquisition Trend", trend)
                                    
                                st.info("📥 **Acquisition Momentum**: Based on when you discover and add new papers")
                        else:
                            st.info("Add more papers over different months to see acquisition patterns")
                    
                    with activity_tabs[2]:
                        st.markdown("##### Reading vs Acquisition Comparison")
                        if reading_activity and acquisition_activity:
                            # Create combined chart showing both activities
                            all_months = sorted(set(list(reading_activity.keys()) + list(acquisition_activity.keys())))
                            
                            reading_values = [reading_activity.get(month, 0) for month in all_months]
                            acquisition_values = [acquisition_activity.get(month, 0) for month in all_months]
                            
                            combined_fig = go.Figure()
                            
                            combined_fig.add_trace(go.Scatter(
                                x=all_months,
                                y=reading_values,
                                mode='lines+markers',
                                name='Papers Read',
                                line=dict(color='#28a745', width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>Papers Read: %{y}<extra></extra>'
                            ))
                            
                            combined_fig.add_trace(go.Scatter(
                                x=all_months,
                                y=acquisition_values,
                                mode='lines+markers',
                                name='Papers Added',
                                line=dict(color='#007bff', width=3),
                                marker=dict(size=8),
                                hovertemplate='<b>%{x}</b><br>Papers Added: %{y}<extra></extra>'
                            ))
                            
                            combined_fig.update_layout(
                                title="Reading Activity vs Paper Acquisition",
                                xaxis_title="Month",
                                yaxis_title="Number of Papers",
                                height=400,
                                hovermode='x unified'
                            )
                            
                            st.plotly_chart(combined_fig, use_container_width=True)
                            
                            # Calculate balance metrics
                            total_read = sum(reading_values)
                            total_added = sum(acquisition_values)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Papers Read", f"{total_read:,}")
                            with col2:
                                st.metric("Total Papers Added", f"{total_added:,}")
                            with col3:
                                if total_added > 0:
                                    efficiency = (total_read / total_added) * 100
                                    st.metric("Reading Efficiency", f"{efficiency:.1f}%")
                                else:
                                    st.metric("Reading Efficiency", "N/A")
                            with col4:
                                backlog = total_added - total_read
                                st.metric("Current Backlog", f"{backlog:,}")
                                
                            # Provide insights based on the comparison
                            if total_read > total_added * 0.8:
                                st.success("🎉 **Excellent balance**: You're keeping up well with your reading!")
                            elif total_read > total_added * 0.5:
                                st.info("📚 **Good progress**: You're making steady progress through your papers")
                            else:
                                st.warning("📈 **Growing backlog**: Consider slowing down acquisition or increasing reading time")
                                
                        else:
                            st.info("Complete some papers with reading dates to see the combined analysis")
                            
                else:
                    # Legacy single activity display (fallback for old data structure)
                    st.markdown("##### Research Activity Over Time")
                    monthly_data = analysis_data["monthly_activity"]
                    
                    # Create interactive line chart
                    months = list(monthly_data.keys())
                    values = list(monthly_data.values())
                    
                    momentum_fig = px.line(
                        x=months, 
                        y=values,
                        title="Research Activity Over Time",
                        labels={'x': 'Month', 'y': 'Papers Added'},
                        markers=True
                    )
                    momentum_fig.update_traces(
                        line=dict(width=3),
                        marker=dict(size=8),
                        hovertemplate='<b>%{x}</b><br>Papers: %{y}<extra></extra>'
                    )
                    momentum_fig.update_layout(height=400)
                    st.plotly_chart(momentum_fig, use_container_width=True)
                    
                    # Calculate and display momentum metrics
                    if len(monthly_data) >= 3:
                        recent_months = list(monthly_data.items())[-3:]
                        earlier_months = list(monthly_data.items())[:-3] if len(monthly_data) > 3 else []
                        
                        recent_avg = sum(month[1] for month in recent_months) / len(recent_months)
                        if earlier_months:
                            earlier_avg = sum(month[1] for month in earlier_months) / len(earlier_months)
                            momentum_change = ((recent_avg - earlier_avg) / max(earlier_avg, 0.1)) * 100
                        else:
                            momentum_change = 0
                            
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Recent Average", f"{recent_avg:.1f} papers/month")
                        with col2:
                            st.metric("Momentum Change", f"{momentum_change:+.1f}%", 
                                    delta=f"{momentum_change:.1f}%")
                        with col3:
                            trend = "📈 Accelerating" if momentum_change > 20 else "📉 Decelerating" if momentum_change < -20 else "➡️ Stable"
                            st.metric("Trend", trend)
                            
                        # Calculate impact metrics to get velocity trend
                        impact_metrics = calculate_research_impact_metrics(st.session_state.paper)
                        st.info(f"🚀 **Research Velocity Trend:** {impact_metrics['research_velocity_trend'].capitalize()}")
                    else:
                        st.info("Add more papers over different months to analyze momentum trends")
            else:
                st.info("Add more papers to track research momentum")
                
        with progress_tabs[3]:
            st.markdown("#### Research Profile Radar Chart")
            if analysis_data["total_papers"] > 0:
                impact_metrics = calculate_research_impact_metrics(st.session_state.paper)
                radar_fig = create_radar_chart(impact_metrics)
                st.plotly_chart(radar_fig, use_container_width=True)
                
                # Add interpretation
                st.markdown("##### Profile Interpretation")
                st.write("Your research profile shows:")
                
                interpretations = []
                if impact_metrics['knowledge_breadth'] > 1.0:
                    interpretations.append("🌐 **Broad Explorer**: You engage with diverse research topics")
                elif impact_metrics['knowledge_breadth'] < 0.5:
                    interpretations.append("🔬 **Deep Specialist**: You focus intensively on specific areas")
                else:
                    interpretations.append("⚖️ **Balanced Researcher**: You maintain good breadth-depth balance")
                    
                if impact_metrics['research_efficiency'] > 0.7:
                    interpretations.append("⚡ **Efficient Reader**: High completion rate indicates good focus")
                elif impact_metrics['research_efficiency'] < 0.3:
                    interpretations.append("📚 **Knowledge Collector**: You acquire more papers than you complete")
                    
                for interpretation in interpretations:
                    st.markdown(interpretation)
            else:
                st.info("Add papers to see your research profile")

        # Display citation network and paper relationships
        st.subheader("🔬 Research Knowledge Graph")
        
        # Create tabs for different knowledge graph views
        knowledge_tabs = st.tabs(["🕸️ Topic Network", "📈 Topic Evolution", "🏆 Top Research Areas"])
        
        with knowledge_tabs[0]:
            st.markdown("#### Research Domain Relationship Network")
            # Topic network visualization with improved academic context
            network_fig = create_topic_network(st.session_state.paper)
            if network_fig:
                network_fig.update_layout(
                    title="Research Domain Relationship Network",
                    height=600,
                    showlegend=True,
                    hovermode='closest'
                )
                st.plotly_chart(network_fig, use_container_width=True)
                
                # Add network statistics
                if analysis_data["topic_distribution"]:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Topics", len(analysis_data["topic_distribution"]))
                    with col2:
                        # Calculate average connections per topic
                        total_connections = sum(1 for topics in st.session_state.paper['Topics'] 
                                              if isinstance(topics, list) and len(topics) > 1)
                        avg_connections = total_connections / max(1, len(analysis_data["topic_distribution"]))
                        st.metric("Avg. Connections", f"{avg_connections:.1f}")
                    with col3:
                        # Most connected topic
                        most_connected = max(analysis_data["topic_distribution"].items(), key=lambda x: x[1])
                        st.metric("Hub Topic", most_connected[0])
                        
                    # Network insights
                    with st.expander("🧠 Network Insights", expanded=False):
                        if len(analysis_data["topic_distribution"]) > 5:
                            st.markdown("🌐 **Rich network**: Your research spans multiple interconnected domains")
                        elif len(analysis_data["topic_distribution"]) > 2:
                            st.markdown("🔗 **Connected research**: Good topic relationships forming")
                        else:
                            st.markdown("🌱 **Growing network**: Add more topics to see connections")
                            
                        if avg_connections > 2:
                            st.markdown("🕸️ **Highly interconnected**: Your research topics frequently overlap")
                        elif avg_connections > 1:
                            st.markdown("🔗 **Well connected**: Good topic integration in your research")
                        else:
                            st.markdown("📍 **Focused research**: Each paper tends to focus on specific topics")
            else:
                st.info("Add more papers with multiple topics to visualize your research domain network")
                st.markdown("**💡 Tip**: Papers with 2+ topics create connections in the network")
                
        with knowledge_tabs[1]:
            st.markdown("#### Topic Evolution Over Time")
            # Topic evolution over time
            topic_evolution_fig = create_topic_evolution(st.session_state.paper)
            if topic_evolution_fig:
                topic_evolution_fig.update_layout(
                    title="Research Focus Evolution",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(topic_evolution_fig, use_container_width=True)
                
                # Add evolution insights
                with st.expander("📊 Evolution Analysis", expanded=False):
                    if len(st.session_state.paper) > 5:
                        # Calculate topic consistency over time
                        recent_papers = st.session_state.paper.tail(len(st.session_state.paper)//3)
                        early_papers = st.session_state.paper.head(len(st.session_state.paper)//3)
                        
                        recent_topics = set()
                        early_topics = set()
                        
                        for topics in recent_papers['Topics']:
                            if isinstance(topics, list):
                                recent_topics.update(topics)
                                
                        for topics in early_papers['Topics']:
                            if isinstance(topics, list):
                                early_topics.update(topics)
                        
                        persistent_topics = recent_topics.intersection(early_topics)
                        new_topics = recent_topics - early_topics
                        abandoned_topics = early_topics - recent_topics
                        
                        if persistent_topics:
                            st.markdown(f"🔄 **Persistent interests**: {', '.join(list(persistent_topics)[:3])}")
                        if new_topics:
                            st.markdown(f"🌱 **Emerging interests**: {', '.join(list(new_topics)[:3])}")
                        if abandoned_topics:
                            st.markdown(f"📉 **Declining interests**: {', '.join(list(abandoned_topics)[:3])}")
                    else:
                        st.info("Add more papers over time to see evolution patterns")
            else:
                st.info("Add more papers with topics and dates to visualize topic evolution")
                st.markdown("**💡 Tip**: Papers with dates show how your interests change over time")
                
        with knowledge_tabs[2]:
            st.markdown("#### Top Research Areas Analysis")
            if analysis_data["topic_distribution"]:
                # Create a detailed analysis of top research areas
                top_topics = dict(sorted(analysis_data["topic_distribution"].items(),
                                       key=lambda x: x[1], reverse=True))
                
                # Create an interactive bar chart for topics
                topics_df = pd.DataFrame([
                    {'Topic': topic, 'Papers': count, 'Percentage': f"{(count/analysis_data['total_papers']*100):.1f}%"}
                    for topic, count in list(top_topics.items())[:10]  # Top 10 topics
                ])
                
                topic_fig = px.bar(
                    topics_df, 
                    x='Papers', 
                    y='Topic',
                    orientation='h',
                    title="Research Areas by Paper Count",
                    text='Percentage',
                    color='Papers',
                    color_continuous_scale='viridis'
                )
                topic_fig.update_traces(
                    hovertemplate='<b>%{y}</b><br>Papers: %{x}<br>Percentage: %{text}<extra></extra>'
                )
                topic_fig.update_layout(
                    height=max(400, len(topics_df) * 40),
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(topic_fig, use_container_width=True)
                
                # Research diversity analysis
                col1, col2 = st.columns(2)
                with col1:
                    # Calculate research diversity index
                    if len(analysis_data["topic_distribution"]) > 1:
                        total = sum(analysis_data["topic_distribution"].values())
                        diversity = -sum((count / total) * np.log(count / total)
                                       for count in analysis_data["topic_distribution"].values())
                        normalized_diversity = diversity / np.log(len(analysis_data["topic_distribution"]))
                        st.metric("Research Diversity Index", f"{normalized_diversity:.3f}",
                                help="0 = very focused, 1 = very diverse")
                        
                        # Diversity interpretation
                        if normalized_diversity > 0.8:
                            st.success("🌐 **Highly diverse**: You explore many different research areas")
                        elif normalized_diversity > 0.5:
                            st.info("⚖️ **Balanced**: Good mix of focus and exploration") 
                        else:
                            st.info("🎯 **Focused**: You concentrate on specific research areas")
                    else:
                        st.metric("Research Diversity Index", "N/A")
                        
                with col2:
                    # Topic specialization analysis
                    if top_topics:
                        top_topic_name, top_topic_count = list(top_topics.items())[0]
                        specialization_ratio = top_topic_count / analysis_data["total_papers"]
                        
                        st.metric("Primary Specialization", f"{specialization_ratio:.1%}",
                                help=f"Percentage of papers in '{top_topic_name}'")
                                
                        if specialization_ratio > 0.5:
                            st.warning(f"🎯 **Highly specialized** in '{top_topic_name}'")
                        elif specialization_ratio > 0.3:
                            st.info(f"🔬 **Focused expertise** in '{top_topic_name}'")
                        else:
                            st.success("🌐 **Well-distributed** research interests")
                            
                # Detailed topic table with enhanced information
                with st.expander("📋 Detailed Topic Statistics", expanded=False):
                    detailed_topics = []
                    for topic, count in top_topics.items():
                        percentage = (count / analysis_data["total_papers"]) * 100
                        
                        # Calculate expertise level
                        if count >= 5:
                            expertise = "Expert 🏆"
                        elif count >= 3:
                            expertise = "Proficient 🥈"
                        elif count >= 2:
                            expertise = "Developing 🥉"
                        else:
                            expertise = "Beginner 🌱"
                            
                        detailed_topics.append({
                            'Topic': topic,
                            'Papers': count,
                            'Percentage': f"{percentage:.1f}%",
                            'Expertise Level': expertise
                        })
                    
                    detailed_df = pd.DataFrame(detailed_topics)
                    st.dataframe(detailed_df, hide_index=True, use_container_width=True)
            else:
                st.info("Add topics to your papers to analyze research domains")
                st.markdown("**💡 Tip**: Add relevant topic tags to your papers to see domain analysis")

        # Research forecast section
        st.subheader("📅 Research Forecast & Planning")
        
        # Create tabs for forecast analysis
        forecast_tabs = st.tabs(["📊 Reading Forecast", "📈 Forecast Metrics", "🎯 Action Plans"])
        
        with forecast_tabs[0]:
            st.markdown("#### 6-Month Research Forecast")
            # Use the provided function to create the reading forecast
            forecast_result = create_reading_forecast(st.session_state.paper)
            if forecast_result and len(forecast_result) == 2:
                forecast_fig, forecast_metrics = forecast_result
                if forecast_fig:
                    forecast_fig.update_layout(
                        title="Research Activity & Backlog Forecast",
                        height=500,
                        hovermode='x unified',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        )
                    )
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    
                    # Add forecast interpretation
                    with st.expander("📊 Forecast Analysis", expanded=True):
                        if forecast_metrics:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Current Situation:**")
                                st.markdown(f"• Backlog: {forecast_metrics.get('current_backlog', 0)} papers")
                                st.markdown(f"• Reading rate: {forecast_metrics.get('read_rate', 0)}% completion")
                                st.markdown(f"• Monthly intake: {forecast_metrics.get('avg_monthly_papers', 0):.1f} papers")
                                
                            with col2:
                                st.markdown("**Forecast:**")
                                clearance_time = forecast_metrics.get('months_to_clear', '∞')
                                if clearance_time != '∞':
                                    st.markdown(f"• Time to clear backlog: {clearance_time} months")
                                else:
                                    st.markdown("• Time to clear backlog: Very long (adjust habits)")
                                    
                                trend = forecast_metrics.get('backlog_trend', 'stable').capitalize()
                                st.markdown(f"• Backlog trend: {trend}")
                                
                                # Provide recommendations based on trend
                                if forecast_metrics.get('backlog_trend') == 'increasing':
                                    st.warning("⚠️ **Action needed**: Backlog is growing")
                                elif forecast_metrics.get('backlog_trend') == 'decreasing':
                                    st.success("✅ **Good progress**: Backlog is shrinking")
                                else:
                                    st.info("➡️ **Stable**: Backlog is manageable")
                        else:
                            st.info("Forecast calculations in progress...")
                else:
                    st.info("Insufficient historical data for accurate forecasting")
            else:
                st.info("Add more papers with dates to generate research forecasts")
                st.markdown("**💡 Tip**: Papers with both addition and reading dates provide better forecasting")
                
        with forecast_tabs[1]:
            st.markdown("#### Detailed Forecast Metrics")
            forecast_result = create_reading_forecast(st.session_state.paper)
            if forecast_result and len(forecast_result) == 2:
                forecast_fig, forecast_metrics = forecast_result
                if forecast_metrics:
                    # Create enhanced metrics display
                    metrics_data = [
                        {
                            'metric': 'Current Backlog',
                            'value': f"{forecast_metrics.get('current_backlog', 0):,}",
                            'description': 'Papers you haven\'t completed yet',
                            'status': 'warning' if forecast_metrics.get('current_backlog', 0) > 10 else 'normal',
                            'emoji': '📚'
                        },
                        {
                            'metric': 'Reading Rate',
                            'value': f"{forecast_metrics.get('read_rate', 0):.1f}%",
                            'description': 'Percentage of papers you typically complete',
                            'status': 'good' if forecast_metrics.get('read_rate', 0) > 50 else 'warning',
                            'emoji': '📖'
                        },
                        {
                            'metric': 'Monthly Intake',
                            'value': f"{forecast_metrics.get('avg_monthly_papers', 0):.1f}",
                            'description': 'Average papers added per month',
                            'status': 'normal',
                            'emoji': '📥'
                        },
                        {
                            'metric': 'Clearance Timeline',
                            'value': f"{forecast_metrics.get('months_to_clear', '∞')} months",
                            'description': 'Time needed to clear current backlog',
                            'status': 'warning' if forecast_metrics.get('months_to_clear') != '∞' and float(str(forecast_metrics.get('months_to_clear', 0)).replace('∞', '999')) > 12 else 'good',
                            'emoji': '⏰'
                        },
                        {
                            'metric': 'Backlog Trend',
                            'value': forecast_metrics.get('backlog_trend', 'stable').capitalize(),
                            'description': 'Direction your backlog is heading',
                            'status': 'good' if forecast_metrics.get('backlog_trend') == 'decreasing' else 'warning',
                            'emoji': '📈' if forecast_metrics.get('backlog_trend') == 'increasing' else '📉' if forecast_metrics.get('backlog_trend') == 'decreasing' else '➡️'
                        }
                    ]
                    
                    for metric_info in metrics_data:
                        with st.expander(f"{metric_info['emoji']} {metric_info['metric']}: {metric_info['value']}", 
                                       expanded=(metric_info['status'] == 'warning')):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                # Color code based on status
                                if metric_info['status'] == 'good':
                                    st.success(metric_info['value'])
                                elif metric_info['status'] == 'warning':
                                    st.warning(metric_info['value'])
                                else:
                                    st.info(metric_info['value'])
                                    
                            with col2:
                                st.write(metric_info['description'])
                else:
                    st.info("Generate forecast first to see detailed metrics")
            else:
                st.info("Add papers with dates to calculate forecast metrics")
                
        with forecast_tabs[2]:
            st.markdown("#### Personalized Action Plans")
            forecast_result = create_reading_forecast(st.session_state.paper)
            if forecast_result and len(forecast_result) == 2:
                forecast_fig, forecast_metrics = forecast_result
                if forecast_metrics:
                    # Generate personalized recommendations based on forecast
                    st.markdown("##### 🎯 Recommendations Based on Your Forecast:")
                    
                    recommendations = []
                    
                    # Backlog management recommendations
                    current_backlog = forecast_metrics.get('current_backlog', 0)
                    if current_backlog > 20:
                        recommendations.append({
                            'priority': 'High',
                            'title': 'Reduce Paper Acquisition',
                            'description': f'With {current_backlog} papers in your backlog, consider pausing new additions until you complete some existing papers.',
                            'actions': [
                                'Set a "no new papers" rule until backlog drops below 15',
                                'Focus on completing 2-3 papers per week',
                                'Review backlog and remove papers no longer relevant'
                            ]
                        })
                    elif current_backlog > 10:
                        recommendations.append({
                            'priority': 'Medium',
                            'title': 'Balance Acquisition and Reading',
                            'description': f'Your backlog of {current_backlog} papers is manageable but needs attention.',
                            'actions': [
                                'For every new paper added, complete one existing paper',
                                'Allocate specific time slots for reading',
                                'Prioritize papers by relevance and impact'
                            ]
                        })
                    
                    # Reading rate recommendations
                    read_rate = forecast_metrics.get('read_rate', 0)
                    if read_rate < 30:
                        recommendations.append({
                            'priority': 'High',
                            'title': 'Improve Reading Completion',
                            'description': f'Your {read_rate:.1f}% completion rate suggests you might be too ambitious with paper selection.',
                            'actions': [
                                'Be more selective when adding papers',
                                'Set reading goals: aim for 50% completion rate',
                                'Use the "15-minute rule": spend 15 minutes scanning before committing'
                            ]
                        })
                    elif read_rate > 80:
                        recommendations.append({
                            'priority': 'Low',
                            'title': 'Consider Expanding Research Scope',
                            'description': f'Your {read_rate:.1f}% completion rate is excellent! You might benefit from exploring more papers.',
                            'actions': [
                                'Gradually increase monthly paper acquisition',
                                'Explore adjacent research areas',
                                'Consider more challenging or comprehensive papers'
                            ]
                        })
                    
                    # Timeline recommendations
                    clearance_months = forecast_metrics.get('months_to_clear', '∞')
                    if clearance_months != '∞' and float(str(clearance_months)) > 12:
                        recommendations.append({
                            'priority': 'Medium',
                            'title': 'Optimize Reading Schedule',
                            'description': f'At current pace, it will take {clearance_months} months to clear your backlog.',
                            'actions': [
                                'Increase daily reading time by 15-30 minutes',
                                'Use techniques like skimming for less critical papers',
                                'Set weekly reading targets',
                                'Consider removing outdated papers from backlog'
                            ]
                        })
                    
                    # Display recommendations
                    if recommendations:
                        for i, rec in enumerate(recommendations):
                            priority_color = {
                                'High': '🔴',
                                'Medium': '🟡', 
                                'Low': '🟢'
                            }
                            
                            with st.expander(f"{priority_color.get(rec['priority'], '⚪')} [{rec['priority']} Priority] {rec['title']}", 
                                           expanded=(rec['priority'] == 'High')):
                                st.markdown(f"**Analysis:** {rec['description']}")
                                st.markdown("**Action Steps:**")
                                for action in rec['actions']:
                                    st.markdown(f"• {action}")
                                    
                                # Add implementation tracker
                                if st.button(f"Mark as Implemented", key=f"implement_{i}"):
                                    st.success("Great! Keep tracking your progress.")
                    else:
                        st.success("🎉 **Excellent balance!** Your reading habits are well-optimized.")
                        st.markdown("**Suggested optimizations:**")
                        st.markdown("• Continue current reading pace")
                        st.markdown("• Periodically review and adjust goals")
                        st.markdown("• Consider exploring new research areas")
                else:
                    st.info("Generate forecast to see personalized action plans")
            else:
                st.info("Add papers with reading history to generate action plans")
        
        # Research insights and recommendations section
        st.subheader("🧠 Advanced Research Analytics")
        
        # Calculate advanced metrics
        if analysis_data["total_papers"] > 2:
            advanced_metrics = calculate_advanced_research_metrics(st.session_state.paper)
            productivity_metrics = calculate_research_productivity_metrics(st.session_state.paper)
            reading_time_metrics = calculate_reading_time_metrics(st.session_state.paper)
            topic_relationships = analyze_topic_relationships(st.session_state.paper)
            
            # Create tabs for different advanced analytics
            advanced_tabs = st.tabs([
                "⏱️ Reading Time Analysis", 
                "🏆 Research Quality Score", 
                "📊 Advanced Metrics", 
                "📈 Comprehensive Dashboard",
                "🔍 Research Insights",
                "⚡ Productivity Analysis",
                "📋 Personalized Recommendations"
            ])
            
            with advanced_tabs[0]:
                # Reading Time Analysis Section
                st.markdown("#### Reading Time & Pattern Analysis")
                if reading_time_metrics and len(reading_time_metrics) > 0:
                    # Create detailed time analysis tabs
                    time_subtabs = st.tabs(["📊 Time Metrics", "📈 Visualizations", "🔍 Pattern Insights"])
                    
                    with time_subtabs[0]:
                        st.markdown("##### Key Reading Time Metrics")
                        
                        # Create metric cards
                        time_metrics_data = [
                            {
                                'name': 'Average Processing Time',
                                'key': 'avg_processing_time_days',
                                'unit': 'days',
                                'description': 'Time from adding a paper to completing it',
                                'good_threshold': 30,
                                'emoji': '⏱️'
                            },
                            {
                                'name': 'Average Publication Lag',
                                'key': 'avg_publication_lag_days', 
                                'unit': 'days',
                                'description': 'Time between paper publication and your reading',
                                'good_threshold': 365,
                                'emoji': '📅'
                            },
                            {
                                'name': 'Reading Velocity',
                                'key': 'reading_velocity_papers_per_month',
                                'unit': 'papers/month',
                                'description': 'Papers completed per month',
                                'good_threshold': 2,
                                'emoji': '🚀'
                            },
                            {
                                'name': 'Average Backlog Age',
                                'key': 'avg_backlog_age_days',
                                'unit': 'days',
                                'description': 'Average age of unread papers',
                                'good_threshold': 90,
                                'emoji': '📚'
                            }
                        ]
                        
                        for metric in time_metrics_data:
                            if metric['key'] in reading_time_metrics:
                                value = reading_time_metrics[metric['key']]
                                
                                with st.expander(f"{metric['emoji']} {metric['name']}: {value:.1f} {metric['unit']}", 
                                               expanded=True):
                                    col1, col2 = st.columns([1, 2])
                                    
                                    with col1:
                                        # Determine status based on threshold
                                        if metric['unit'] == 'papers/month':
                                            status = 'good' if value >= metric['good_threshold'] else 'needs_improvement'
                                        else:
                                            status = 'good' if value <= metric['good_threshold'] else 'needs_improvement'
                                            
                                        if status == 'good':
                                            st.success(f"{value:.1f} {metric['unit']}")
                                        else:
                                            st.warning(f"{value:.1f} {metric['unit']}")
                                            
                                    with col2:
                                        st.write(metric['description'])
                                        
                                        # Add contextual advice
                                        if metric['key'] == 'avg_processing_time_days':
                                            if value > 60:
                                                st.info("💡 Consider setting reading deadlines to improve processing speed")
                                            elif value < 7:
                                                st.success("⚡ Excellent quick processing - you read papers promptly")
                                        elif metric['key'] == 'avg_publication_lag_days':
                                            if value > 730:  # 2 years
                                                st.info("📚 You tend to read established papers - good for foundational knowledge")
                                            elif value < 180:  # 6 months
                                                st.success("🔥 You stay current with recent research")
                                        elif metric['key'] == 'reading_velocity_papers_per_month':
                                            if value > 10:
                                                st.info("🚀 High velocity - ensure you're retaining information")
                                            elif value < 1:
                                                st.info("🐢 Steady pace - quality over quantity approach")
                            else:
                                st.info(f"{metric['emoji']} {metric['name']}: Complete more papers to see this metric")
                    
                    with time_subtabs[1]:
                        st.markdown("##### Reading Time Visualizations")
                        reading_time_viz = create_reading_time_visualizations(st.session_state.paper, reading_time_metrics)
                        
                        if reading_time_viz:
                            viz_options = []
                            if 'processing_time_hist' in reading_time_viz:
                                viz_options.append("Processing Time Distribution")
                            if 'reading_timeline' in reading_time_viz:
                                viz_options.append("Reading Timeline")
                            if 'publication_lag' in reading_time_viz:
                                viz_options.append("Publication Lag Analysis")
                            if 'monthly_velocity' in reading_time_viz:
                                viz_options.append("Monthly Reading Velocity")
                            if 'efficiency_trend' in reading_time_viz:
                                viz_options.append("Reading Efficiency Trend")
                                
                            if viz_options:
                                selected_viz = st.selectbox("Select Visualization:", viz_options)
                                
                                if selected_viz == "Processing Time Distribution" and 'processing_time_hist' in reading_time_viz:
                                    st.plotly_chart(reading_time_viz['processing_time_hist'], use_container_width=True)
                                    st.info("📊 This shows how long it typically takes you to read papers after adding them")
                                    
                                elif selected_viz == "Reading Timeline" and 'reading_timeline' in reading_time_viz:
                                    st.plotly_chart(reading_time_viz['reading_timeline'], use_container_width=True)
                                    st.info("📈 Timeline of your reading activity over time")
                                    
                                elif selected_viz == "Publication Lag Analysis" and 'publication_lag' in reading_time_viz:
                                    st.plotly_chart(reading_time_viz['publication_lag'], use_container_width=True)
                                    st.info("📅 Analysis of how current vs historical your reading is")
                                    
                                elif selected_viz == "Monthly Reading Velocity" and 'monthly_velocity' in reading_time_viz:
                                    st.plotly_chart(reading_time_viz['monthly_velocity'], use_container_width=True)
                                    st.info("🚀 Your reading speed trends over different months")
                                    
                                elif selected_viz == "Reading Efficiency Trend" and 'efficiency_trend' in reading_time_viz:
                                    st.plotly_chart(reading_time_viz['efficiency_trend'], use_container_width=True)
                                    st.info("⚡ How your reading efficiency changes over time")
                            else:
                                st.info("Complete more papers to unlock visualizations")
                        else:
                            st.info("Add papers with completion dates to see reading time visualizations")
                    
                    with time_subtabs[2]:
                        st.markdown("##### Reading Pattern Insights")
                        if len(reading_time_metrics) > 4:
                            insight_cards = []
                            
                            if 'most_productive_day' in reading_time_metrics:
                                insight_cards.append({
                                    'title': '📅 Most Productive Day',
                                    'content': f"You tend to complete most papers on **{reading_time_metrics['most_productive_day']}**",
                                    'type': 'info'
                                })
                            
                            if 'reading_trend' in reading_time_metrics:
                                trend_emojis = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}
                                insight_cards.append({
                                    'title': f"{trend_emojis.get(reading_time_metrics['reading_trend'], '➡️')} Reading Trend",
                                    'content': f"Your reading activity is **{reading_time_metrics['reading_trend']}**",
                                    'type': 'success' if reading_time_metrics['reading_trend'] == 'increasing' else 'info'
                                })
                            
                            if 'reading_consistency_score' in reading_time_metrics:
                                consistency = reading_time_metrics['reading_consistency_score']
                                insight_cards.append({
                                    'title': '🎯 Reading Consistency',
                                    'content': f"Your reading consistency score is **{consistency:.2f}** (0=irregular, 1=very consistent)",
                                    'type': 'success' if consistency > 0.7 else 'warning' if consistency < 0.3 else 'info'
                                })
                            
                            if 'reading_recency_score' in reading_time_metrics:
                                recency = reading_time_metrics['reading_recency_score']
                                insight_cards.append({
                                    'title': '🆕 Reading Recency',
                                    'content': f"Your recency score is **{recency:.2f}** - {'you prefer recent papers' if recency > 0.5 else 'you read papers regardless of age'}",
                                    'type': 'info'
                                })
                            
                            # Display insight cards
                            for card in insight_cards:
                                with st.expander(card['title'], expanded=True):
                                    if card['type'] == 'success':
                                        st.success(card['content'])
                                    elif card['type'] == 'warning':
                                        st.warning(card['content'])
                                    else:
                                        st.info(card['content'])
                        else:
                            st.info("Complete more papers with dates to unlock pattern insights")
                else:
                    st.info("Add papers with reading completion dates to see detailed reading time analysis")
                    st.markdown("**💡 Tips to unlock this analysis:**")
                    st.markdown("- Add 'Date Read' when marking papers as completed")
                    st.markdown("- Include 'Date Published' for publication lag analysis")
                    st.markdown("- Complete at least 3-5 papers for meaningful patterns")
            
            with advanced_tabs[1]:
                # Research Quality Score
                st.markdown("#### Research Quality Assessment")
                quality_score, score_breakdown = calculate_research_quality_score(
                    st.session_state.paper, analysis_data, advanced_metrics, productivity_metrics
                )
                
                # Main quality score display
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    # Create an enhanced gauge chart for the quality score
                    gauge_fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = quality_score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Research Quality Score", 'font': {'size': 24}},
                        delta = {'reference': 75, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkblue", 'thickness': 0.8},
                            'steps': [
                                {'range': [0, 25], 'color': "#ffcccc"},
                                {'range': [25, 50], 'color': "#ffffcc"}, 
                                {'range': [50, 75], 'color': "#ccffcc"},
                                {'range': [75, 90], 'color': "#ccffff"},
                                {'range': [90, 100], 'color': "#ccccff"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 85
                            }
                        }
                    ))
                    
                    gauge_fig.update_layout(height=400, font={'size': 16})
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Grade display with interpretation
                    grade = score_breakdown['grade']
                    grade_colors = {'A+': '🏆', 'A': '🥇', 'B+': '🥈', 'B': '🥉', 'C+': '📊', 'C': '📈', 'D': '📉', 'F': '⚠️'}
                    st.markdown(f"### {grade_colors.get(grade, '📊')} Grade: {grade}")
                
                # Detailed score breakdown
                st.markdown("#### Score Components Breakdown")
                components = score_breakdown['components']
                
                # Create interactive bar chart for components
                components_df = pd.DataFrame([
                    {'Component': comp.replace('_', ' ').title(), 'Score': score} 
                    for comp, score in components.items()
                ])
                
                comp_fig = px.bar(
                    components_df, 
                    x='Component', 
                    y='Score', 
                    title="Research Quality Components",
                    color='Score', 
                    color_continuous_scale='RdYlGn',
                    text='Score'
                )
                comp_fig.update_traces(
                    texttemplate='%{text:.1f}',
                    textposition='outside',
                    hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}/100<extra></extra>'
                )
                comp_fig.update_layout(
                    height=400,
                    xaxis_tickangle=-45,
                    showlegend=False
                )
                st.plotly_chart(comp_fig, use_container_width=True)
                
                # Detailed component analysis
                st.markdown("#### Component Analysis & Recommendations")
                
                component_details = {
                    'completion_rate': {
                        'name': 'Completion Rate',
                        'description': 'Percentage of papers you\'ve finished reading',
                        'optimal': '>70%',
                        'emoji': '✅'
                    },
                    'diversity_score': {
                        'name': 'Research Diversity',
                        'description': 'How diverse your research interests are',
                        'optimal': '0.5-0.8',
                        'emoji': '🌐'
                    },
                    'consistency_score': {
                        'name': 'Reading Consistency',
                        'description': 'Regularity of your reading habits',
                        'optimal': '>0.6',
                        'emoji': '📊'
                    },
                    'velocity_score': {
                        'name': 'Reading Velocity',
                        'description': 'Speed of reading papers',
                        'optimal': '2-5 papers/month',
                        'emoji': '🚀'
                    },
                    'recency_score': {
                        'name': 'Research Currency',
                        'description': 'How current your research reading is',
                        'optimal': '>0.5',
                        'emoji': '🆕'
                    }
                }
                
                for comp_key, comp_score in components.items():
                    if comp_key in component_details:
                        detail = component_details[comp_key]
                        
                        with st.expander(f"{detail['emoji']} {detail['name']}: {comp_score:.1f}/100", 
                                       expanded=(comp_score < 50)):  # Expand low scores
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Status indicator
                                if comp_score >= 80:
                                    st.success(f"{comp_score:.1f}/100")
                                    status = "Excellent"
                                elif comp_score >= 60:
                                    st.info(f"{comp_score:.1f}/100")
                                    status = "Good"
                                elif comp_score >= 40:
                                    st.warning(f"{comp_score:.1f}/100")
                                    status = "Needs Improvement"
                                else:
                                    st.error(f"{comp_score:.1f}/100")
                                    status = "Critical"
                                    
                                st.caption(f"Status: {status}")
                                st.caption(f"Optimal: {detail['optimal']}")
                                
                            with col2:
                                st.write(detail['description'])
                                
                                # Component-specific recommendations
                                if comp_key == 'completion_rate' and comp_score < 60:
                                    st.markdown("**💡 Improvement Tips:**")
                                    st.markdown("• Be more selective when adding papers")
                                    st.markdown("• Set reading deadlines for yourself")
                                    st.markdown("• Focus on completing papers before adding new ones")
                                elif comp_key == 'diversity_score':
                                    if comp_score < 40:
                                        st.markdown("**💡 Improvement Tips:**")
                                        st.markdown("• Explore adjacent research areas")
                                        st.markdown("• Follow interdisciplinary journals")
                                        st.markdown("• Attend conferences outside your main field")
                                    elif comp_score > 80:
                                        st.markdown("**💡 Focus Tips:**")
                                        st.markdown("• Consider specializing in 2-3 core areas")
                                        st.markdown("• Look for connections between your diverse interests")
                                elif comp_key == 'consistency_score' and comp_score < 50:
                                    st.markdown("**💡 Improvement Tips:**")
                                    st.markdown("• Set a regular reading schedule")
                                    st.markdown("• Use reading goals and tracking")
                                    st.markdown("• Block time in calendar for research reading")
                                elif comp_key == 'velocity_score':
                                    if comp_score < 50:
                                        st.markdown("**💡 Speed Tips:**")
                                        st.markdown("• Use skimming techniques for initial screening")
                                        st.markdown("• Focus on abstract and conclusions first")
                                        st.markdown("• Set time limits for each paper")
                                    elif comp_score > 80:
                                        st.markdown("**💡 Quality Tips:**")
                                        st.markdown("• Ensure you're retaining key information")
                                        st.markdown("• Take more detailed notes")
                                        st.markdown("• Consider reading fewer but more impactful papers")
                                elif comp_key == 'recency_score' and comp_score < 50:
                                    st.markdown("**💡 Currency Tips:**")
                                    st.markdown("• Subscribe to recent paper alerts")
                                    st.markdown("• Follow leading researchers on social media")
                                    st.markdown("• Balance recent papers with foundational work")
                
                # Overall recommendations
                st.markdown("#### Overall Assessment & Action Plan")
                
                if quality_score >= 85:
                    st.success("🏆 **Outstanding Research Habits!** You're in the top tier of research practices.")
                    st.markdown("**Continue to:**")
                    st.markdown("• Maintain your excellent reading discipline")
                    st.markdown("• Consider mentoring others in research practices")
                    st.markdown("• Share your successful strategies")
                    
                elif quality_score >= 70:
                    st.success("🥈 **Strong Research Practices!** You have solid research habits with room for optimization.")
                    st.markdown("**Focus on:**")
                    lowest_component = min(components.items(), key=lambda x: x[1])
                    st.markdown(f"• Improving your {lowest_component[0].replace('_', ' ').title()} (current: {lowest_component[1]:.1f})")
                    st.markdown("• Maintaining your current strengths")
                    
                elif quality_score >= 50:
                    st.warning("🥉 **Developing Research Skills** - You're making progress but there's significant room for improvement.")
                    low_components = [k for k, v in components.items() if v < 60]
                    st.markdown("**Priority improvements:**")
                    for comp in low_components[:3]:  # Top 3 priorities
                        st.markdown(f"• {comp.replace('_', ' ').title()}")
                        
                else:
                    st.error("📈 **Research Habits Need Attention** - Focus on building fundamental research practices.")
                    st.markdown("**Immediate actions:**")
                    st.markdown("• Set a consistent reading schedule")
                    st.markdown("• Complete existing papers before adding new ones")
                    st.markdown("• Start with 1-2 papers per week goal")
            
            with advanced_tabs[2]:
                # Advanced metrics display - one by one with detailed explanations
                st.markdown("#### Advanced Research Metrics Deep Dive")
                
                # Create expandable sections for each advanced metric
                advanced_metric_sections = [
                    {
                        'name': 'Shannon Diversity Index',
                        'value': advanced_metrics.get('shannon_diversity', 0),
                        'description': 'Measures the diversity and evenness of your research topics',
                        'interpretation': {
                            'high': (0.8, 'Very diverse research interests - you explore many different areas'),
                            'medium': (0.4, 'Balanced research approach - good mix of focus and exploration'),
                            'low': (0.0, 'Focused research interests - deep specialization in specific areas')
                        },
                        'emoji': '🌐'
                    },
                    {
                        'name': 'Reading Consistency',
                        'value': advanced_metrics.get('reading_consistency', 0),
                        'description': 'Measures how consistent your reading habits are over time',
                        'interpretation': {
                            'high': (0.7, 'Very consistent reading habits - regular research engagement'),
                            'medium': (0.4, 'Moderately consistent - some variability in reading patterns'),
                            'low': (0.0, 'Irregular reading patterns - sporadic research engagement')
                        },
                        'emoji': '📊'
                    },
                    {
                        'name': 'Research Momentum',
                        'value': advanced_metrics.get('research_momentum', 'stable'),
                        'description': 'Direction and speed of your research activity changes',
                        'interpretation': {
                            'accelerating': 'Research activity is increasing - growing engagement',
                            'stable': 'Consistent research pace - steady progress',
                            'decelerating': 'Research activity is declining - may need motivation boost'
                        },
                        'emoji': '🚀'
                    },
                    {
                        'name': 'Topic Clustering Coefficient',
                        'value': advanced_metrics.get('topic_clustering', 0),
                        'description': 'Measures how interconnected your research topics are',
                        'interpretation': {
                            'high': (0.6, 'Highly interconnected research - papers often span multiple topics'),
                            'medium': (0.3, 'Moderately connected research - some topic overlap'),
                            'low': (0.0, 'Discrete research areas - papers tend to focus on single topics')
                        },
                        'emoji': '🔗'
                    }
                ]
                
                for metric in advanced_metric_sections:
                    if isinstance(metric['value'], str):
                        # Handle non-numeric metrics like research_momentum
                        with st.expander(f"{metric['emoji']} {metric['name']}: {metric['value'].title()}", expanded=True):
                            st.write(metric['description'])
                            if metric['value'] in metric['interpretation']:
                                st.info(f"**Interpretation:** {metric['interpretation'][metric['value']]}")
                    else:
                        # Handle numeric metrics
                        with st.expander(f"{metric['emoji']} {metric['name']}: {metric['value']:.3f}", expanded=True):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.metric(metric['name'], f"{metric['value']:.3f}")
                                
                                # Create a simple progress bar based on the value
                                if metric['value'] <= 1.0:
                                    st.progress(metric['value'])
                                else:
                                    st.progress(min(metric['value'] / 5.0, 1.0))
                                    
                            with col2:
                                st.write(metric['description'])
                                
                                # Provide interpretation based on thresholds
                                for level, (threshold, interpretation) in metric['interpretation'].items():
                                    if level == 'high' and metric['value'] >= threshold:
                                        st.success(f"**{level.title()}:** {interpretation}")
                                        break
                                    elif level == 'medium' and metric['value'] >= threshold:
                                        st.info(f"**{level.title()}:** {interpretation}")
                                        break
                                    elif level == 'low':
                                        st.info(f"**{level.title()}:** {interpretation}")
                                        break
                
                # Domain expertise analysis
                st.markdown("#### Research Domain Expertise")
                domain_expertise = advanced_metrics.get('domain_expertise', {})
                if domain_expertise:
                    expertise_df = pd.DataFrame([
                        {
                            'Topic': topic, 
                            'Level': level, 
                            'Papers': len([p for p in st.session_state.paper['Topics'] 
                                         if isinstance(p, list) and topic in p])
                        }
                        for topic, level in domain_expertise.items()
                    ])
                    
                    # Create an interactive expertise level chart
                    fig = px.bar(
                        expertise_df, 
                        x='Topic', 
                        y='Papers', 
                        color='Level',
                        color_discrete_map={
                            'Expert': '#4CAF50', 
                            'Proficient': '#FFC107', 
                            'Developing': '#FF9800', 
                            'Beginner': '#9E9E9E'
                        },
                        title="Research Domain Expertise Levels",
                        text='Papers'
                    )
                    fig.update_traces(
                        texttemplate='%{text}',
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Level: %{color}<br>Papers: %{y}<extra></extra>'
                    )
                    fig.update_layout(
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed expertise table
                    st.markdown("##### Detailed Expertise Analysis")
                    expertise_analysis = []
                    for _, row in expertise_df.iterrows():
                        topic = row['Topic']
                        level = row['Level']
                        papers = row['Papers']
                        
                        if level == 'Expert':
                            recommendation = "Share your expertise, consider reviewing papers in this area"
                        elif level == 'Proficient':
                            recommendation = "Good foundation, read more advanced papers to become expert"
                        elif level == 'Developing':
                            recommendation = "Keep reading, look for survey papers and foundational work"
                        else:
                            recommendation = "Explore more papers to build understanding"
                            
                        expertise_analysis.append({
                            'Topic': topic,
                            'Level': level,
                            'Papers': papers,
                            'Next Steps': recommendation
                        })
                    
                    expertise_df_detailed = pd.DataFrame(expertise_analysis)
                    st.dataframe(expertise_df_detailed, hide_index=True, use_container_width=True)
                else:
                    st.info("Add topics to your papers to analyze domain expertise")
                    
            with advanced_tabs[3]:
                # Comprehensive Research Dashboard
                st.markdown("#### Comprehensive Research Dashboard")
                dashboard_fig = create_comprehensive_research_dashboard(
                    st.session_state.paper, analysis_data, advanced_metrics, productivity_metrics
                )
                if dashboard_fig:
                    st.plotly_chart(dashboard_fig, use_container_width=True)
                    
                    # Add dashboard interpretation
                    with st.expander("📊 Dashboard Interpretation Guide", expanded=False):
                        st.markdown("**How to read this dashboard:**")
                        st.markdown("• **Top Left**: Paper status distribution over time")
                        st.markdown("• **Top Right**: Reading velocity trends")  
                        st.markdown("• **Bottom Left**: Topic diversity evolution")
                        st.markdown("• **Bottom Right**: Productivity metrics correlation")
                        st.markdown("")
                        st.markdown("**Look for:**")
                        st.markdown("• Consistent patterns in reading habits")
                        st.markdown("• Periods of high/low activity")
                        st.markdown("• Correlation between adding and completing papers")
                        st.markdown("• Topic diversity trends over time")
                else:
                    st.info("Dashboard requires more papers to generate meaningful visualizations")
                    
            with advanced_tabs[4]:
                # Comprehensive Text Insights
                st.markdown("#### AI-Generated Research Pattern Insights")
                text_insights = generate_research_insights_text(analysis_data, advanced_metrics, productivity_metrics)
                
                if text_insights and len(text_insights) > 0:
                    st.markdown("##### 🔍 Key Insights About Your Research Patterns")
                    
                    for i, insight in enumerate(text_insights[:8]):  # Show top 8 insights
                        # Extract insight category from content
                        if ':' in insight:
                            category, content = insight.split(':', 1)
                            title = f"Insight {i+1}: {category.strip()}"
                        else:
                            title = f"Research Pattern Insight {i+1}"
                            content = insight
                            
                        with st.expander(title, expanded=(i < 3)):  # Expand first 3
                            st.markdown(content.strip())
                            
                            # Add action buttons for insights
                            col1, col2 = st.columns([1, 1])
                            with col1:
                                if st.button(f"✅ Helpful", key=f"helpful_{i}"):
                                    st.success("Thanks for the feedback!")
                            with col2:
                                if st.button(f"💡 Want More Info", key=f"more_info_{i}"):
                                    st.info("Consider exploring the Advanced Analytics tabs for deeper analysis.")
                else:
                    st.info("Add more papers to generate AI insights about your research patterns")
                    st.markdown("**💡 Tips to unlock insights:**")
                    st.markdown("• Add at least 5-10 papers")
                    st.markdown("• Include diverse topics")
                    st.markdown("• Mark papers as read when completed")
                    st.markdown("• Add publication and reading dates")
                    
            with advanced_tabs[5]:
                # Productivity metrics - detailed analysis
                st.markdown("#### Research Productivity Deep Analysis")
                
                # Create productivity metric sections
                productivity_sections = [
                    {
                        'title': 'Research Activity Metrics',
                        'metrics': [
                            ('Papers per Month', 'papers_per_month', 'papers/month'),
                            ('Research Span', 'research_span_days', 'days'),
                            ('Reading Velocity', 'reading_velocity_papers_per_month', 'papers/month')
                        ]
                    },
                    {
                        'title': 'Reading Efficiency Metrics',
                        'metrics': [
                            ('Completion Rate', 'completion_rate', '%'),
                            ('Active Reading Rate', 'active_reading_rate', '%'),
                            ('Reading Efficiency', 'reading_efficiency', 'score')
                        ]
                    },
                    {
                        'title': 'Backlog Management Metrics',
                        'metrics': [
                            ('Backlog Ratio', 'backlog_ratio', '%'),
                            ('Estimated Clearance Time', 'estimated_backlog_clearance_months', 'months')
                        ]
                    }
                ]
                
                for section in productivity_sections:
                    with st.expander(f"📊 {section['title']}", expanded=True):
                        cols = st.columns(len(section['metrics']))
                        
                        for i, (name, key, unit) in enumerate(section['metrics']):
                            with cols[i]:
                                if key in productivity_metrics:
                                    value = productivity_metrics[key]
                                    
                                    # Format the value based on unit
                                    if unit == '%':
                                        display_value = f"{value:.1%}" if isinstance(value, float) else f"{value}%"
                                        progress_value = value if isinstance(value, float) else value / 100
                                    elif unit == 'months':
                                        if value == float('inf'):
                                            display_value = "Very Long"
                                            progress_value = 1.0
                                        else:
                                            display_value = f"{value:.1f} {unit}"
                                            progress_value = min(value / 12, 1.0)  # Scale to 12 months
                                    else:
                                        display_value = f"{value:.1f} {unit}"
                                        if 'per_month' in key:
                                            progress_value = min(value / 10, 1.0)  # Scale to 10 papers/month
                                        else:
                                            progress_value = min(value / 365, 1.0)  # Scale to 1 year for days
                                    
                                    st.metric(name, display_value)
                                    
                                    # Add progress bar for visual indication
                                    if unit == '%':
                                        if progress_value > 0.7:
                                            st.success("Excellent")
                                        elif progress_value > 0.4:
                                            st.info("Good")
                                        else:
                                            st.warning("Needs Improvement")
                                    else:
                                        st.progress(progress_value)
                                else:
                                    st.metric(name, "N/A")
                                    st.caption("More data needed")
                
                # Productivity trends visualization
                st.markdown("##### Productivity Trends")
                if 'reading_efficiency_trend' in advanced_metrics and advanced_metrics['reading_efficiency_trend']:
                    trend_data = advanced_metrics['reading_efficiency_trend']
                    
                    # Create efficiency trend chart
                    trend_df = pd.DataFrame(trend_data)
                    
                    efficiency_fig = px.line(
                        trend_df, 
                        x='period', 
                        y='efficiency',
                        title="Reading Efficiency Over Time",
                        markers=True
                    )
                    efficiency_fig.update_traces(
                        line=dict(width=3),
                        marker=dict(size=10),
                        hovertemplate='<b>%{x}</b><br>Efficiency: %{y:.1%}<br>Papers Added: %{customdata[0]}<br>Papers Completed: %{customdata[1]}<extra></extra>',
                        customdata=[[row['papers_added'], row['papers_completed']] for row in trend_data]
                    )
                    efficiency_fig.update_layout(
                        height=400,
                        yaxis=dict(tickformat='.0%')
                    )
                    st.plotly_chart(efficiency_fig, use_container_width=True)
                else:
                    st.info("Add papers over multiple months to see productivity trends")
                    
            with advanced_tabs[6]:
                # Personalized recommendations
                st.markdown("#### Personalized Research Recommendations")
                recommendations = generate_research_recommendations(
                    st.session_state.paper, advanced_metrics, productivity_metrics
                )
                
                if recommendations and len(recommendations) > 0:
                    st.markdown("##### 🎯 AI-Generated Recommendations for Your Research")
                    
                    # Group recommendations by priority
                    priority_groups = {'High': [], 'Medium': [], 'Low': []}
                    for rec in recommendations:
                        if rec['priority'] in priority_groups:
                            priority_groups[rec['priority']].append(rec)
                    
                    # Display by priority
                    for priority in ['High', 'Medium', 'Low']:
                        if priority_groups[priority]:
                            priority_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                            st.markdown(f"##### {priority_colors[priority]} {priority} Priority Recommendations")
                            
                            for i, rec in enumerate(priority_groups[priority]):
                                with st.expander(f"{rec['title']}", expanded=(priority == 'High')):
                                    st.markdown(f"**Category:** {rec['category']}")
                                    st.markdown(f"**Analysis:** {rec['description']}")
                                    
                                    st.markdown("**📋 Actionable Steps:**")
                                    for j, step in enumerate(rec['actionable_steps'], 1):
                                        st.markdown(f"{j}. {step}")
                                    
                                    # Add implementation tracking
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        if st.button(f"✅ Implemented", key=f"impl_{priority}_{i}"):
                                            st.success("Great progress! Keep it up!")
                                    with col2:
                                        if st.button(f"📅 Plan to Implement", key=f"plan_{priority}_{i}"):
                                            st.info("Added to your action plan!")
                                    with col3:
                                        if st.button(f"ℹ️ More Details", key=f"details_{priority}_{i}"):
                                            st.info("Check the other Advanced Analytics tabs for deeper insights.")
                                            
                else:
                    st.info("Add more papers to receive personalized recommendations")
                    st.markdown("**💡 To unlock recommendations:**")
                    st.markdown("• Add at least 5-10 papers")
                    st.markdown("• Include reading status and dates")
                    st.markdown("• Add diverse topics")
                    st.markdown("• Complete some papers to establish patterns")
            
            # Advanced visualizations - show one at a time with detailed analysis
            st.markdown("### 📈 Advanced Research Visualizations")
            advanced_viz = create_advanced_visualizations(st.session_state.paper, advanced_metrics, topic_relationships)
            reading_time_viz = create_reading_time_visualizations(st.session_state.paper, reading_time_metrics)
            
            # Create tabs for different visualization categories
            viz_main_tabs = st.tabs([
                "⏰ Time-Based Analysis", 
                "📊 Activity Patterns", 
                "🌐 Topic Analysis", 
                "🔗 Network Analysis",
                "🎯 Domain Expertise"
            ])
            
            with viz_main_tabs[0]:
                # Time-based visualizations
                st.markdown("#### Time-Based Research Analysis")
                if reading_time_viz:
                    time_viz_options = []
                    if 'processing_time_hist' in reading_time_viz:
                        time_viz_options.append("Processing Time Distribution")
                    if 'reading_timeline' in reading_time_viz:
                        time_viz_options.append("Reading Timeline")
                    if 'publication_lag' in reading_time_viz:
                        time_viz_options.append("Publication Lag Analysis")
                    if 'monthly_velocity' in reading_time_viz:
                        time_viz_options.append("Monthly Reading Velocity")
                    if 'efficiency_trend' in reading_time_viz:
                        time_viz_options.append("Reading Efficiency Trend")
                        
                    if time_viz_options:
                        selected_time_viz = st.selectbox("📊 Select Time Analysis:", time_viz_options)
                        
                        if selected_time_viz == "Processing Time Distribution" and 'processing_time_hist' in reading_time_viz:
                            st.markdown("##### 📊 How Long You Take to Read Papers After Adding Them")
                            reading_time_viz['processing_time_hist'].update_layout(height=500)
                            st.plotly_chart(reading_time_viz['processing_time_hist'], use_container_width=True)
                            
                            with st.expander("📖 How to interpret this chart", expanded=False):
                                st.markdown("**What it shows:** Distribution of time between adding and completing papers")
                                st.markdown("**Ideal pattern:** Most papers completed within 30-60 days")
                                st.markdown("**Red flags:** Many papers taking >90 days indicates backlog issues")
                                st.markdown("**Green flags:** Consistent processing times indicate good habits")
                                
                        elif selected_time_viz == "Reading Timeline" and 'reading_timeline' in reading_time_viz:
                            st.markdown("##### 📈 Your Reading Activity Over Time")
                            reading_time_viz['reading_timeline'].update_layout(height=500)
                            st.plotly_chart(reading_time_viz['reading_timeline'], use_container_width=True)
                            
                            with st.expander("📖 Timeline Analysis Guide", expanded=False):
                                st.markdown("**What it shows:** When you complete papers vs when you add them")
                                st.markdown("**Look for:** Gaps between adding and reading peaks")
                                st.markdown("**Ideal pattern:** Steady reading pace that matches adding pace")
                                st.markdown("**Action items:** Identify periods of high additions but low completions")
                                
                        elif selected_time_viz == "Publication Lag Analysis" and 'publication_lag' in reading_time_viz:
                            st.markdown("##### 📅 How Current vs Historical Your Reading Is")
                            reading_time_viz['publication_lag'].update_layout(height=500)
                            st.plotly_chart(reading_time_viz['publication_lag'], use_container_width=True)
                            
                            with st.expander("📅 Publication Currency Guide", expanded=False):
                                st.markdown("**What it shows:** Time between paper publication and your reading")
                                st.markdown("**Recent research:** <1 year lag keeps you current")
                                st.markdown("**Foundational research:** 2-5 year lag is normal for established knowledge")
                                st.markdown("**Historical research:** >5 years may indicate foundational learning")
                                
                        elif selected_time_viz == "Monthly Reading Velocity" and 'monthly_velocity' in reading_time_viz:
                            st.markdown("##### 🚀 Reading Speed Patterns by Month")
                            reading_time_viz['monthly_velocity'].update_layout(height=500)
                            st.plotly_chart(reading_time_viz['monthly_velocity'], use_container_width=True)
                            
                            with st.expander("🚀 Velocity Analysis Guide", expanded=False):
                                st.markdown("**What it shows:** Papers completed per month over time")
                                st.markdown("**Seasonal patterns:** Look for academic year influences")
                                st.markdown("**Target velocity:** 2-5 papers/month for most researchers")
                                st.markdown("**Optimization:** Identify your peak performance periods")
                                
                        elif selected_time_viz == "Reading Efficiency Trend" and 'efficiency_trend' in reading_time_viz:
                            st.markdown("##### ⚡ How Your Reading Efficiency Changes Over Time")
                            reading_time_viz['efficiency_trend'].update_layout(height=500)
                            st.plotly_chart(reading_time_viz['efficiency_trend'], use_container_width=True)
                            
                            with st.expander("⚡ Efficiency Optimization Guide", expanded=False):
                                st.markdown("**What it shows:** Percentage of added papers that get completed each month")
                                st.markdown("**Target efficiency:** 60-80% is excellent")
                                st.markdown("**Declining trend:** May indicate backlog accumulation")
                                st.markdown("**Improving trend:** Shows better reading discipline")
                    else:
                        st.info("Complete more papers with dates to unlock time-based visualizations")
                else:
                    st.info("Add papers with completion dates to see time-based analysis")
                    
            with viz_main_tabs[1]:
                # Activity patterns
                st.markdown("#### Research Activity Patterns")
                if 'productivity_heatmap' in advanced_viz:
                    st.markdown("##### 📊 Research Activity Heatmap")
                    advanced_viz['productivity_heatmap'].update_layout(height=500)
                    st.plotly_chart(advanced_viz['productivity_heatmap'], use_container_width=True)
                    
                    with st.expander("🔥 Heatmap Analysis Guide", expanded=False):
                        st.markdown("**What it shows:** Days/times when you're most active in research")
                        st.markdown("**Hot spots:** Identify your most productive periods")
                        st.markdown("**Cold spots:** Times when you might want to schedule research")
                        st.markdown("**Patterns:** Look for weekly or monthly cycles")
                        
                    # Add activity statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Peak Activity Day", "Analysis based on heatmap")
                    with col2:
                        st.metric("Most Active Time", "Check heatmap colors")
                    with col3:
                        st.metric("Activity Consistency", "Pattern regularity")
                else:
                    st.info("Add more papers with varied dates to generate productivity heatmap")
                    
            with viz_main_tabs[2]:
                # Topic analysis
                st.markdown("#### Topic Evolution & Relationships")
                if 'topic_evolution' in advanced_viz:
                    st.markdown("##### 📈 How Your Research Focus Evolves")
                    advanced_viz['topic_evolution'].update_layout(height=500)
                    st.plotly_chart(advanced_viz['topic_evolution'], use_container_width=True)
                    
                    with st.expander("🔄 Evolution Interpretation", expanded=False):
                        st.markdown("**What it shows:** How your interest in different topics changes over time")
                        st.markdown("**Rising topics:** Growing interest areas - potential specializations")
                        st.markdown("**Declining topics:** Decreasing focus - completed exploration or shifting interests")
                        st.markdown("**Stable topics:** Core research areas - consistent focus")
                        
                    # Topic evolution insights
                    if 'focus_evolution' in advanced_metrics:
                        focus_evo = advanced_metrics['focus_evolution']
                        if isinstance(focus_evo, dict):
                            col1, col2 = st.columns(2)
                            with col1:
                                if 'emerging_topics' in focus_evo and focus_evo['emerging_topics']:
                                    st.markdown("**🌱 Emerging Topics:**")
                                    for topic in focus_evo['emerging_topics'][:5]:
                                        st.markdown(f"• {topic}")
                                        
                            with col2:
                                if 'declining_topics' in focus_evo and focus_evo['declining_topics']:
                                    st.markdown("**📉 Declining Topics:**")
                                    for topic in focus_evo['declining_topics'][:5]:
                                        st.markdown(f"• {topic}")
                else:
                    st.info("Add more papers over time to see topic evolution")
                    
            with viz_main_tabs[3]:
                # Network analysis
                st.markdown("#### Research Network Analysis")
                if 'enhanced_network' in advanced_viz:
                    st.markdown("##### 🕸️ Enhanced Topic Relationship Network")
                    advanced_viz['enhanced_network'].update_layout(height=600)
                    st.plotly_chart(advanced_viz['enhanced_network'], use_container_width=True)
                    
                    with st.expander("🕸️ Network Analysis Guide", expanded=False):
                        st.markdown("**Node size:** Represents how many papers include that topic")
                        st.markdown("**Edge thickness:** Shows how often topics appear together")
                        st.markdown("**Clusters:** Groups of closely related topics")
                        st.markdown("**Central nodes:** Core topics that connect to many others")
                        st.markdown("**Isolated nodes:** Specialized topics with few connections")
                        
                    # Network statistics
                    if 'topic_associations' in topic_relationships and topic_relationships['topic_associations']:
                        st.markdown("##### 🔗 Topic Relationship Insights")
                        associations = topic_relationships['topic_associations']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**🏆 Most Connected Topics:**")
                            if isinstance(associations, dict):
                                # Find topics with most connections
                                topic_connections = {topic: len(assoc_list) for topic, assoc_list in associations.items() 
                                                   if isinstance(assoc_list, list)}
                                top_connected = sorted(topic_connections.items(), key=lambda x: x[1], reverse=True)[:5]
                                
                                for topic, connections in top_connected:
                                    st.markdown(f"• **{topic}**: {connections} connections")
                                    
                        with col2:
                            st.markdown("**🤝 Strongest Associations:**")
                            if isinstance(associations, dict):
                                for topic, assoc_list in list(associations.items())[:3]:
                                    if assoc_list and isinstance(assoc_list, list):
                                        top_assoc = assoc_list[0] if len(assoc_list) > 0 else None
                                        if top_assoc and len(top_assoc) >= 2:
                                            st.markdown(f"• **{topic}** ↔ **{top_assoc[0]}**")
                else:
                    st.info("Add papers with multiple topics to see enhanced network")
                    
            with viz_main_tabs[4]:
                # Domain expertise visualization
                st.markdown("#### Domain Expertise Analysis")
                domain_expertise = advanced_metrics.get('domain_expertise', {})
                if domain_expertise:
                    # Create expertise progression chart
                    expertise_df = pd.DataFrame([
                        {
                            'Topic': topic, 
                            'Level': level, 
                            'Papers': len([p for p in st.session_state.paper['Topics'] 
                                         if isinstance(p, list) and topic in p]),
                            'Level_Numeric': {'Beginner': 1, 'Developing': 2, 'Proficient': 3, 'Expert': 4}.get(level, 1)
                        }
                        for topic, level in domain_expertise.items()
                    ])
                    
                    # Expertise level progression
                    st.markdown("##### 🎯 Research Domain Expertise Progression")
                    
                    # Create a scatter plot showing papers vs expertise level
                    expertise_fig = px.scatter(
                        expertise_df, 
                        x='Papers', 
                        y='Level_Numeric',
                        size='Papers',
                        color='Level',
                        hover_name='Topic',
                        color_discrete_map={
                            'Expert': '#4CAF50', 
                            'Proficient': '#FFC107', 
                            'Developing': '#FF9800', 
                            'Beginner': '#9E9E9E'
                        },
                        title="Expertise Level vs Paper Count by Topic"
                    )
                    expertise_fig.update_layout(
                        height=500,
                        yaxis=dict(
                            tickmode='array',
                            tickvals=[1, 2, 3, 4],
                            ticktext=['Beginner', 'Developing', 'Proficient', 'Expert']
                        ),
                        xaxis_title="Number of Papers",
                        yaxis_title="Expertise Level"
                    )
                    expertise_fig.update_traces(
                        hovertemplate='<b>%{hovertext}</b><br>Papers: %{x}<br>Level: %{color}<extra></extra>'
                    )
                    st.plotly_chart(expertise_fig, use_container_width=True)
                    
                    # Expertise development roadmap
                    st.markdown("##### 🗺️ Expertise Development Roadmap")
                    
                    # Group by expertise level
                    expertise_levels = {'Beginner': [], 'Developing': [], 'Proficient': [], 'Expert': []}
                    for _, row in expertise_df.iterrows():
                        expertise_levels[row['Level']].append((row['Topic'], row['Papers']))
                    
                    roadmap_cols = st.columns(4)
                    level_colors = ['🌱', '📚', '🎓', '🏆']
                    
                    for i, (level, topics) in enumerate(expertise_levels.items()):
                        with roadmap_cols[i]:
                            st.markdown(f"#### {level_colors[i]} {level}")
                            if topics:
                                for topic, papers in topics[:3]:  # Show top 3 per level
                                    st.markdown(f"**{topic}** ({papers} papers)")
                                    
                                    # Next step recommendations
                                    if level == 'Beginner':
                                        st.caption("📖 Read 2-3 more papers")
                                    elif level == 'Developing':
                                        st.caption("📚 Find survey papers")
                                    elif level == 'Proficient':
                                        st.caption("🔬 Read cutting-edge research")
                                    else:
                                        st.caption("✍️ Consider contributing")
                            else:
                                st.info("No topics at this level yet")
                                
                    with st.expander("🚀 Expertise Development Tips", expanded=False):
                        st.markdown("**Beginner → Developing (2-3 papers):**")
                        st.markdown("• Read foundational papers and surveys")
                        st.markdown("• Focus on understanding key concepts")
                        st.markdown("")
                        st.markdown("**Developing → Proficient (4-5 papers):**")
                        st.markdown("• Read recent conference/journal papers")
                        st.markdown("• Compare different approaches")
                        st.markdown("")
                        st.markdown("**Proficient → Expert (6+ papers):**")
                        st.markdown("• Read cutting-edge research")
                        st.markdown("• Consider contributing reviews or papers")
                        st.markdown("• Mentor others in this area")
                else:
                    st.info("Add topics to your papers to analyze domain expertise")
            
        else:
            st.info("Add at least 3 papers to unlock advanced research analytics")
            st.markdown("**💡 What you'll unlock with more papers:**")
            st.markdown("• AI-powered reading time analysis")
            st.markdown("• Research quality scoring")
            st.markdown("• Topic evolution tracking")
            st.markdown("• Personalized productivity recommendations")
            st.markdown("• Advanced visualizations and insights")
            
        # Basic Research Insights & Recommendations section (for smaller collections)
        st.subheader("🧠 Basic Research Insights & Recommendations")

        # Use direct insights generation for basic analysis
        if analysis_data["total_papers"] > 0:
            # Create insights tabs
            insight_tabs = st.tabs(["🎯 Research Focus", "📚 Reading Habits", "📅 Research Planning"])

            with insight_tabs[0]:
                st.markdown("##### Research Focus Analysis")
                focus_insights = []
                
                # Calculate basic impact metrics for insights
                if analysis_data["total_papers"] > 0:
                    basic_impact_metrics = calculate_research_impact_metrics(st.session_state.paper)
                    
                    # Topic diversity insights
                    if basic_impact_metrics["knowledge_breadth"] < 0.3 and analysis_data["total_papers"] > 5:
                        focus_insights.append(
                            "🔍 **Low topic diversity** - Consider exploring adjacent research areas to broaden your knowledge base and discover new connections.")
                    elif basic_impact_metrics["knowledge_breadth"] > 1.5:
                        focus_insights.append(
                            "🌐 **Very high topic diversity** - Consider focusing on fewer areas for deeper expertise while maintaining your broad perspective.")

                    # Topic concentration insights
                    if basic_impact_metrics["topic_concentration"] > 0.5 and analysis_data["topic_distribution"]:
                        top_topic = max(analysis_data["topic_distribution"].items(), key=lambda x: x[1])[0]
                        focus_insights.append(
                            f"🎯 **Strong specialization in '{top_topic}'** - Your focused approach may lead to deep expertise. Consider exploring how this topic connects to related areas.")

                if analysis_data["topic_distribution"]:
                    if len(analysis_data["topic_distribution"]) < 3 and analysis_data["total_papers"] > 5:
                        focus_insights.append(
                            "🔬 **Limited research domains** - Consider exploring 2-3 related research areas to enhance interdisciplinary understanding and create new research opportunities.")

                # Display insights
                if focus_insights:
                    for i, insight in enumerate(focus_insights):
                        with st.expander(f"Focus Insight {i+1}", expanded=True):
                            st.markdown(insight)
                else:
                    st.info("Add more papers with diverse topics to generate research focus insights.")
                    st.markdown("**💡 Tips to get focus insights:**")
                    st.markdown("• Add papers from 3+ different research areas")
                    st.markdown("• Include topic tags for each paper")
                    st.markdown("• Aim for 5+ papers to see meaningful patterns")

            with insight_tabs[1]:
                st.markdown("##### Reading Habits Analysis")
                reading_insights = []

                # Reading progress insights
                completion_rate = analysis_data["completion_rate"]
                if completion_rate < 30:
                    reading_insights.append({
                        'title': 'Low Completion Rate',
                        'insight': f"📉 Your completion rate is {completion_rate:.1f}%. Focus on completing existing papers before adding new ones to build momentum and avoid backlog overwhelm.",
                        'actions': [
                            'Set a goal to complete 2-3 papers before adding new ones',
                            'Use the "15-minute rule" - read for 15 minutes to get started',
                            'Remove papers that no longer seem relevant'
                        ]
                    })
                elif completion_rate > 70:
                    reading_insights.append({
                        'title': 'High Completion Rate',
                        'insight': f"📈 Excellent completion rate of {completion_rate:.1f}%! You're effectively completing papers. Consider expanding your research scope or tackling more challenging papers.",
                        'actions': [
                            'Gradually increase your monthly paper goal',
                            'Explore more complex or comprehensive papers',
                            'Consider papers outside your immediate area'
                        ]
                    })

                # Reading velocity insights
                velocity = analysis_data["reading_velocity"]
                if velocity < 1:
                    reading_insights.append({
                        'title': 'Methodical Reading Pace',
                        'insight': f"🐢 Your reading pace of {velocity:.1f} papers/month suggests thorough analysis, which is excellent for deep understanding and retention.",
                        'actions': [
                            'Continue your thorough approach',
                            'Take detailed notes to maximize retention',
                            'Consider increasing pace slightly if desired'
                        ]
                    })
                elif velocity > 5:
                    reading_insights.append({
                        'title': 'High Reading Velocity',
                        'insight': f"🚀 High reading velocity of {velocity:.1f} papers/month! Ensure you're retaining key information from your rapid reading.",
                        'actions': [
                            'Create summary notes for each paper',
                            'Test your retention with periodic reviews',
                            'Balance speed with comprehension quality'
                        ]
                    })

                # Display insights with actionable recommendations
                if reading_insights:
                    for insight_data in reading_insights:
                        with st.expander(f"📊 {insight_data['title']}", expanded=True):
                            st.markdown(insight_data['insight'])
                            st.markdown("**🎯 Recommended Actions:**")
                            for action in insight_data['actions']:
                                st.markdown(f"• {action}")
                else:
                    st.info("Continue adding and reading papers to generate reading habit insights.")
                    st.markdown("**💡 Tips to get reading insights:**")
                    st.markdown("• Mark papers as 'Read' when completed")
                    st.markdown("• Add papers consistently over time")
                    st.markdown("• Track your reading for at least 2-3 months")

            with insight_tabs[2]:
                st.markdown("##### Research Planning Insights")
                planning_insights = []
                
                # Get forecast data for planning insights
                forecast_result = create_reading_forecast(st.session_state.paper)
                if forecast_result and len(forecast_result) == 2:
                    forecast_fig, forecast_metrics = forecast_result
                    
                    if forecast_metrics:
                        # Backlog insights
                        current_backlog = forecast_metrics.get('current_backlog', 0)
                        if current_backlog > 10:
                            planning_insights.append({
                                'title': 'Backlog Management',
                                'insight': f"📚 You have {current_backlog} unread papers. This backlog needs strategic management to prevent overwhelm.",
                                'actions': [
                                    f'Prioritize your top {min(5, current_backlog)} most important papers',
                                    'Set aside dedicated reading time each week',
                                    'Consider removing papers that are no longer relevant'
                                ],
                                'type': 'warning'
                            })

                        # Timeline insights
                        clearance_time = forecast_metrics.get('months_to_clear', '∞')
                        if clearance_time != '∞':
                            try:
                                clearance_float = float(clearance_time)
                                if clearance_float > 6:
                                    planning_insights.append({
                                        'title': 'Reading Timeline',
                                        'insight': f"⏳ At your current pace, it will take {clearance_time} months to clear your backlog. Consider optimizing your reading strategy.",
                                        'actions': [
                                            'Increase weekly reading time by 2-3 hours',
                                            'Use skimming techniques for less critical papers',
                                            'Set monthly reading goals',
                                            'Be more selective with new paper additions'
                                        ],
                                        'type': 'info'
                                    })
                            except (ValueError, TypeError):
                                pass

                        # Trend insights
                        backlog_trend = forecast_metrics.get('backlog_trend', 'stable')
                        if backlog_trend == 'increasing':
                            planning_insights.append({
                                'title': 'Growing Backlog Trend',
                                'insight': "📈 Your backlog is growing, indicating you're adding papers faster than reading them.",
                                'actions': [
                                    'Implement a "one in, one out" policy',
                                    'Schedule regular reading sessions',
                                    'Review and prune your reading list monthly'
                                ],
                                'type': 'warning'
                            })
                        elif backlog_trend == 'decreasing':
                            planning_insights.append({
                                'title': 'Improving Backlog Management',
                                'insight': "📉 Excellent! Your backlog is shrinking, showing good reading discipline.",
                                'actions': [
                                    'Maintain your current reading pace',
                                    'Consider gradually increasing paper discovery',
                                    'Share your successful strategies with others'
                                ],
                                'type': 'success'
                            })

                # Display planning insights
                if planning_insights:
                    for insight_data in planning_insights:
                        with st.expander(f"📅 {insight_data['title']}", expanded=True):
                            if insight_data['type'] == 'warning':
                                st.warning(insight_data['insight'])
                            elif insight_data['type'] == 'success':
                                st.success(insight_data['insight'])
                            else:
                                st.info(insight_data['insight'])
                                
                            st.markdown("**📋 Action Plan:**")
                            for action in insight_data['actions']:
                                st.markdown(f"• {action}")
                else:
                    st.info("Add more papers over time to generate research planning insights.")
                    st.markdown("**💡 Tips to get planning insights:**")
                    st.markdown("• Add papers consistently over several weeks")
                    st.markdown("• Mark reading status accurately")
                    st.markdown("• Include dates when adding papers")

        # NEW ARXIV-SPECIFIC ANALYSIS SECTION
        st.subheader("🔬 ArXiv Research Analysis")
        
        arxiv_tabs = st.tabs(["📊 ArXiv Statistics", "👥 Author Analysis", "📚 Category Distribution", "📅 Publication Timeline"])
        
        with arxiv_tabs[0]:
            st.markdown("#### ArXiv Engagement Metrics")
            arxiv_stats = analysis_data.get('arxiv_statistics', {})
            
            if arxiv_stats.get('arxiv_papers', 0) > 0:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("ArXiv Papers", f"{arxiv_stats['arxiv_papers']}")
                    st.metric("ArXiv Engagement", f"{arxiv_stats['arxiv_percentage']:.1f}%")
                    
                with col2:
                    st.metric("PDF Available", f"{arxiv_stats['pdf_available']}")
                    st.metric("PDF Availability", f"{arxiv_stats['pdf_percentage']:.1f}%")
                    
                with col3:
                    # Version statistics
                    version_dist = analysis_data.get('version_distribution', {})
                    if version_dist:
                        st.metric("Updated Papers", f"{version_dist.get('updated_papers_percentage', 0):.1f}%")
                    else:
                        st.metric("Updated Papers", "N/A")
                        
                # ArXiv engagement insights
                if arxiv_stats['arxiv_percentage'] > 70:
                    st.success("🌟 **High ArXiv Engagement**: You're very connected to cutting-edge research!")
                elif arxiv_stats['arxiv_percentage'] > 40:
                    st.info("📈 **Good ArXiv Engagement**: Solid connection to recent research developments")
                else:
                    st.info("📚 **Mixed Sources**: You use diverse paper sources beyond ArXiv")
                    
            else:
                st.info("No ArXiv papers detected. Add ArXiv papers or run auto-fill to see ArXiv-specific metrics.")
        
        with arxiv_tabs[1]:
            st.markdown("#### Author Engagement Analysis")
            author_engagement = analysis_data.get('author_engagement', {})
            
            if author_engagement.get('most_read_authors'):
                # Display top authors
                st.markdown("##### 🏆 Most Read Authors")
                top_authors = list(author_engagement['most_read_authors'].items())[:10]
                
                # Create a horizontal bar chart for top authors
                author_names = [author for author, _ in top_authors]
                author_counts = [count for _, count in top_authors]
                
                author_fig = px.bar(
                    x=author_counts, 
                    y=author_names,
                    orientation='h',
                    title="Top Authors by Paper Count",
                    labels={'x': 'Number of Papers', 'y': 'Author'},
                    color=author_counts,
                    color_continuous_scale='Blues'
                )
                author_fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(author_fig, use_container_width=True)
                
                # Author engagement metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Unique Authors", author_engagement.get('unique_authors', 0))
                with col2:
                    st.metric("Avg Authors/Paper", f"{author_engagement.get('avg_authors_per_paper', 0):.1f}")
                    
            else:
                st.info("No author information available. Run auto-fill to extract author data from ArXiv papers.")
                
        with arxiv_tabs[2]:
            st.markdown("#### Research Category Analysis")
            category_dist = analysis_data.get('category_distribution', {})
            
            if category_dist:
                if 'primary_categories' in category_dist and category_dist['primary_categories']:
                    st.markdown("##### 📊 Primary ArXiv Categories")
                    primary_cats = category_dist['primary_categories']
                    
                    # Create pie chart for primary categories
                    cat_fig = px.pie(
                        names=list(primary_cats.keys()),
                        values=list(primary_cats.values()),
                        title="Distribution by Primary ArXiv Category"
                    )
                    cat_fig.update_traces(
                        textposition='inside',
                        textinfo='percent+label',
                        hovertemplate='<b>%{label}</b><br>Papers: %{value}<br>%{percent}<extra></extra>'
                    )
                    st.plotly_chart(cat_fig, use_container_width=True)
                    
                if 'all_categories' in category_dist and category_dist['all_categories']:
                    st.markdown("##### 🎨 All ArXiv Categories")
                    all_cats = category_dist['all_categories']
                    
                    # Create horizontal bar chart for all categories
                    cat_names = list(all_cats.keys())[:15]  # Top 15
                    cat_counts = list(all_cats.values())[:15]
                    
                    all_cat_fig = px.bar(
                        x=cat_counts,
                        y=cat_names,
                        orientation='h',
                        title="Papers by All ArXiv Categories (Top 15)",
                        labels={'x': 'Number of Papers', 'y': 'Category'},
                        color=cat_counts,
                        color_continuous_scale='Viridis'
                    )
                    all_cat_fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(all_cat_fig, use_container_width=True)
            else:
                st.info("No category information available. Run auto-fill to extract category data from ArXiv papers.")
                
        with arxiv_tabs[3]:
            st.markdown("#### Publication Timeline Analysis")
            pub_timeline = analysis_data.get('publication_timeline', {})
            
            if pub_timeline and 'papers_by_year' in pub_timeline:
                # Publication timeline visualization
                years = list(pub_timeline['papers_by_year'].keys())
                counts = list(pub_timeline['papers_by_year'].values())
                
                timeline_fig = px.bar(
                    x=years,
                    y=counts,
                    title="Papers by Publication Year",
                    labels={'x': 'Publication Year', 'y': 'Number of Papers'},
                    color=counts,
                    color_continuous_scale='plasma'
                )
                timeline_fig.update_layout(height=400)
                st.plotly_chart(timeline_fig, use_container_width=True)
                
                # Publication metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    if pub_timeline.get('earliest_paper'):
                        st.metric("Earliest Paper", str(pub_timeline['earliest_paper']))
                with col2:
                    if pub_timeline.get('latest_paper'):
                        st.metric("Latest Paper", str(pub_timeline['latest_paper']))
                with col3:
                    if pub_timeline.get('avg_paper_age_years'):
                        st.metric("Avg Paper Age", f"{pub_timeline['avg_paper_age_years']:.1f} years")
                        
                # Publication recency insights
                if pub_timeline.get('avg_paper_age_years', 0) < 2:
                    st.success("🚀 **Very Current**: You read very recent research!")
                elif pub_timeline.get('avg_paper_age_years', 0) < 5:
                    st.info("📅 **Recent Focus**: Good balance of recent and established research")
                else:
                    st.info("📚 **Historical Perspective**: You include foundational and historical papers")
            else:
                st.info("No publication date information available. Run auto-fill to extract publication dates from ArXiv papers.")

    # New Auto-Fill Tab
    with tab4:
        st.markdown("# 🔧 Auto-Fill Missing Information")
        st.markdown("---")
        
        if 'paper' not in st.session_state or st.session_state.paper.empty:
            st.info("📝 No papers found! Add some papers first to use the auto-fill feature.")
        else:
            # Count papers that need updating
            paper_df = st.session_state.paper.copy()
            arxiv_papers = []
            papers_needing_update = []
            
            for index, paper in paper_df.iterrows():
                if is_arxiv_url(paper.get('Link')):
                    arxiv_papers.append(paper)
                    needs_upd, missing_fields = needs_paper_update(paper)
                    if needs_upd:
                        papers_needing_update.append((paper, missing_fields))
            
            st.markdown("### 📊 Current Status")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Papers", len(paper_df))
            with col2:
                st.metric("ArXiv Papers", len(arxiv_papers))
            with col3:
                st.metric("Need Updates", len(papers_needing_update))
            with col4:
                completion_rate = ((len(arxiv_papers) - len(papers_needing_update)) / max(1, len(arxiv_papers))) * 100
                st.metric("Completion Rate", f"{completion_rate:.1f}%")
            
            st.markdown("---")
            
            if not papers_needing_update:
                st.success("🎉 All your ArXiv papers already have complete information!")
            else:
                st.markdown("### 🔍 Papers That Can Be Enhanced")
                
                st.info(f"Found **{len(papers_needing_update)}** ArXiv papers that can be enhanced with additional information from the ArXiv API.")
                
                # Show what information can be auto-filled
                st.markdown("#### Available Information to Auto-Fill:")
                info_cols = st.columns(2)
                with info_cols[0]:
                    st.markdown("""
                    **📚 Paper Details:**
                    • **Authors** - Complete author list
                    • **PDF Link** - Direct PDF download
                    • **Version** - Paper version (v1, v2, etc.)
                    • **Comment** - Pages, appendix info, etc.
                    """)
                with info_cols[1]:
                    st.markdown("""
                    **🏷️ Categories & Dates:**
                    • **Primary Category** - Main subject classification
                    • **All Categories** - Complete category list
                    • **Date Published** - Original publication date
                    • **Date Updated** - Last update on ArXiv
                    """)
                
                # Show preview of some papers that need updating
                if st.checkbox("🔍 Show preview of papers to be updated", value=False):
                    st.markdown("#### Preview (First 5 papers):")
                    for i, (paper, missing_fields) in enumerate(papers_needing_update[:5]):
                        with st.expander(f"📄 {paper['Title'][:60]}{'...' if len(paper['Title']) > 60 else ''}"):
                            st.write(f"**Link:** {paper['Link']}")
                            st.write(f"**Missing Fields:** {', '.join(missing_fields)}")
                
                st.markdown("---")
                
                # Warning about API calls
                st.warning("""
                ⚠️ **Important Notes:**
                • This will make API calls to ArXiv for each paper needing updates
                • The process respects ArXiv's API limits with delays between requests
                • Estimated time: ~30 seconds for 10 papers
                • Your existing data will be preserved - only missing fields will be filled
                """)
                
                # Auto-fill button
                if st.button("🚀 Start Auto-Fill Process", type="primary", use_container_width=True):
                    with st.spinner("🔄 Auto-filling missing paper information..."):
                        updated_count, failed_count = auto_fill_missing_paper_info()
                    
                    if updated_count > 0:
                        st.success(f"🎉 Successfully updated {updated_count} papers!")
                        st.rerun()


if __name__ == "__main__":
    main()