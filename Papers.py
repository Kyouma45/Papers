import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

PAPER_FILE = "paper_with_topics.json"


def load_paper():
    if os.path.exists(PAPER_FILE):
        with open(PAPER_FILE, 'r') as f:
            paper_dict = json.load(f)
            # Ensure Topics is always a list
            for paper in paper_dict:
                if 'Topics' in paper:
                    if isinstance(paper['Topics'], str):
                        paper['Topics'] = [t.strip() for t in paper['Topics'].split(',') if t.strip()]
                    elif paper['Topics'] is None or paper['Topics'] == "":
                        paper['Topics'] = []
                else:
                    paper['Topics'] = []
                # Initialize Description field if not present
                if 'Description' not in paper:
                    paper['Description'] = ""

            paper_df = pd.DataFrame(paper_dict)
            if 'Date Added' not in paper_df.columns:
                paper_df['Date Added'] = pd.NaT
            if 'Link' not in paper_df.columns:
                paper_df['Link'] = ""
            if 'Description' not in paper_df.columns:
                paper_df['Description'] = ""
            return paper_df
    return pd.DataFrame(columns=['Title', 'Reading Status', 'Date Added', 'Link', 'Topics', 'Description'])


def save_paper(paper_df):
    if 'Date Added' in paper_df.columns:
        paper_df['Date Added'] = paper_df['Date Added'].astype(str)
    paper_dict = paper_df.to_dict('records')
    with open(PAPER_FILE, 'w') as f:
        json.dump(paper_dict, f)


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
    # Ensure topics is a list
    topics_list = topics if isinstance(topics, list) else []
    new_paper = pd.DataFrame({
        'Title': [title],
        'Reading Status': [reading_status],
        'Date Added': [date_added if date_added else ""],
        'Link': [link],
        'Topics': [topics_list],
        'Description': [description]
    })
    st.session_state.paper = pd.concat([st.session_state.paper, new_paper], ignore_index=True)
    st.session_state.paper = st.session_state.paper.sort_values('Title')
    save_paper(st.session_state.paper)
    reset_form()


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


def main():
    st.set_page_config(page_title="📚 My Reading Journey", layout="wide")
    st.title("📚 My Reading Journey")

    if 'paper' not in st.session_state:
        st.session_state.paper = load_paper()

    if 'form_reset' not in st.session_state:
        st.session_state.form_reset = True

    tab1, tab2, tab3 = st.tabs(["➕ Add Paper", "📋 View Lists", "🔍 Search & Edit"])

    with tab1:
        st.header("Add New Paper")
        with st.form("add_paper_form", border=True):
            title = st.text_input("📕 Paper Title", placeholder="Enter paper title")
            col1, col2 = st.columns(2)
            with col1:
                date_added = st.date_input("📅 Date Added", datetime.today(), format="YYYY-MM-DD")
                reading_status = st.selectbox("📖 Reading Status", ["Want to Read", "Reading", "Read"])
            with col2:
                link = st.text_input("🌐 Link to Paper (optional)", value="https://arxiv.org/abs/")
                topics = st.text_input("🏷️ Topics", placeholder="e.g., AI, Machine Learning")

            description = st.text_area("📝 Description (optional)",
                                       placeholder="Enter a brief description or notes about the paper")

            submitted = st.form_submit_button("Add Paper", use_container_width=True)
            if submitted and title:
                if title in st.session_state.paper['Title'].values:
                    st.error("🚫 This paper already exists!")
                else:
                    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
                    add_paper(title, reading_status, date_added.strftime('%Y-%m-%d') if date_added else "", link,
                              topic_list, description)
                    st.success("✨ Paper added successfully!")

    with tab2:
        st.header("View Papers")

        # Simplified filter options in two columns
        col1, col2 = st.columns(2)
        with col1:
            all_topics = set(topic for topics_list in st.session_state.paper['Topics'] for topic in topics_list if
                             isinstance(topics_list, list))
            topic_filter = st.multiselect("🏷️ Filter by Topic", options=sorted(list(all_topics)))

        with col2:
            status_filter = st.radio("📖 Filter by Status", ["All", "Want to Read", "Reading", "Read"], horizontal=True)

        papers_to_display = st.session_state.paper.copy()
        papers_to_display['Topics'] = papers_to_display['Topics'].apply(display_topics)

        if status_filter != "All":
            papers_to_display = papers_to_display[papers_to_display['Reading Status'] == status_filter]

        if topic_filter:
            papers_to_display = papers_to_display[
                papers_to_display['Topics'].str.contains('|'.join(topic_filter), case=False, na=False)]

        # Display papers with description
        for _, row in papers_to_display.iterrows():
            with st.expander(f"📑 {row['Title']}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write("**Status:** ", row['Reading Status'])
                    st.write("**Date Added:** ", row['Date Added'])
                    if row['Link']:
                        st.write("**Link:** ", f"[Open Paper]({row['Link']})")
                    st.write("**Topics:** ", row['Topics'])
                with col2:
                    if row['Description']:
                        st.write("**Description:**")
                        st.write(row['Description'])

        # Statistics Section
        st.divider()
        st.subheader("📊 Reading Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Papers", len(papers_to_display))
            status_counts = papers_to_display['Reading Status'].value_counts()
            st.write("📚 Reading Status Breakdown:")
            for status, count in status_counts.items():
                st.write(f"- {status}: {count}")

        with col2:
            all_displayed_topics = [topic.strip() for topics in papers_to_display['Topics'].str.split(',') for topic in
                                    topics if topic.strip()]
            topic_counts = pd.Series(all_displayed_topics).value_counts()

            if not topic_counts.empty:
                st.write("🏷️ Most Common Topics:")
                for topic, count in topic_counts.head(5).items():
                    st.write(f"- {topic}: {count}")
            else:
                st.write("🏷️ No topics found in current selection")

        with col3:
            if 'Date Added' in papers_to_display.columns and not papers_to_display['Date Added'].empty:
                papers_to_display['Date Added'] = pd.to_datetime(papers_to_display['Date Added'])
                most_recent = papers_to_display['Date Added'].max()
                oldest = papers_to_display['Date Added'].min()

                st.write("📅 Time Statistics:")
                st.write(f"- Most recent: {most_recent.strftime('%Y-%m-%d')}")
                st.write(f"- Oldest: {oldest.strftime('%Y-%m-%d')}")

                monthly_counts = papers_to_display.set_index('Date Added').resample('M').size()
                if len(monthly_counts) > 0:
                    avg_papers_per_month = monthly_counts.mean()
                    st.write(f"- Avg papers/month: {avg_papers_per_month:.1f}")

    with tab3:
        st.header("Search & Edit Papers")
        search_term = st.text_input("🔍 Search by Title, Topic, or Description")
        if search_term:
            mask = (st.session_state.paper['Title'].str.contains(search_term, case=False)) | \
                   (st.session_state.paper['Topics'].apply(
                       lambda x: any(search_term.lower() in t.lower() for t in x if isinstance(x, list)))) | \
                   (st.session_state.paper['Description'].str.contains(search_term, case=False, na=False))
            search_results = st.session_state.paper[mask]

            if search_results.empty:
                st.warning("No matching papers found.")
            else:
                for index, row in search_results.iterrows():
                    with st.expander(f"Edit: {row['Title']}"):
                        new_title = st.text_input("Edit Title", value=row['Title'])
                        new_status = st.selectbox(
                            "Edit Status",
                            ["Want to Read", "Reading", "Read"],
                            index=["Want to Read", "Reading", "Read"].index(row['Reading Status']),
                            key=f"status_{index}"
                        )
                        new_date = st.date_input("Edit Date Added", value=pd.to_datetime(row['Date Added']))
                        new_link = st.text_input("Edit Link", value=row['Link'])
                        if row['Link']:
                            st.markdown(f"[Open current link in new tab]({row['Link']})")
                        new_topics = st.text_area("Edit Topics", value=", ".join(row['Topics']), key=f"topics_{index}")
                        new_description = st.text_area("Edit Description", value=row['Description'],
                                                       key=f"description_{index}")

                        if st.button("Update", key=f"update_{index}"):
                            edit_paper(index, new_title, new_status, new_date.strftime('%Y-%m-%d'), new_link,
                                       new_topics, new_description)
                            st.success("✨ Paper updated successfully!")


if __name__ == "__main__":
    main()