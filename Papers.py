import streamlit as st
import pandas as pd
import json
import os
import re  # Added the missing import
from datetime import datetime
import random
import arxiv

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

    tab1, tab2, tab3 = st.tabs(["➕ Add Paper", "📋 View Collection", "🔍 Search & Edit"])

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

        # Enhanced filters
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            all_topics = set(topic for topics_list in st.session_state.paper['Topics'] for topic in topics_list if
                             isinstance(topics_list, list))
            topic_filter = st.multiselect("🏷️ Filter by Topics", options=sorted(list(all_topics)))

        with col2:
            status_filter = st.radio("📖 Reading Status", ["All", "Want to Read", "Reading", "Read"], horizontal=True)

        with col3:
            sort_by = st.selectbox("🔄 Sort by", ["Date Added", "Title"])

        papers_to_display = st.session_state.paper.copy()

        # Apply filters and sorting
        if status_filter != "All":
            papers_to_display = papers_to_display[papers_to_display['Reading Status'] == status_filter]

        if topic_filter:
            papers_to_display = papers_to_display[
                papers_to_display['Topics'].apply(lambda x: any(topic in x for topic in topic_filter))
            ]

        if sort_by == "Date Added":
            papers_to_display = papers_to_display.sort_values('Date Added', ascending=False)
        else:
            papers_to_display = papers_to_display.sort_values('Title')

        # Display papers with formatted date
        for _, paper in papers_to_display.iterrows():
            with st.expander(f"📑 {paper['Title']}", expanded=False):
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown(f"**Status:** {paper['Reading Status']}")
                    # Format the date to show only YYYY-MM-DD
                    date_str = pd.to_datetime(paper['Date Added']).strftime('%Y-%m-%d')
                    st.markdown(f"**Added:** {date_str}")
                    if paper['Link']:
                        st.markdown(f"[Open Paper]({paper['Link']})")
                    st.markdown(f"**Topics:** {', '.join(paper['Topics'])}")

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

if __name__ == "__main__":
    main()
