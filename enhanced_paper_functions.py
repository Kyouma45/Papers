"""
Enhanced Functions for Auto-filling Paper Details

This module contains enhanced functions that can be integrated into Papers.py
to automatically fill missing information for existing papers.
"""

import json
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import urllib3

# Suppress SSL warning messages
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_enhanced_paper_details(arxiv_url):
    """
    Enhanced version of fetch_paper_details that extracts all available information
    from the ArXiv API including authors, categories, version info, etc.
    """
    try:
        # Extract arXiv ID from URL
        arxiv_id = extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            return None, "Invalid arXiv URL format"
        
        # Make request to arXiv API
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(api_url, verify=False, timeout=15)
        
        if response.status_code != 200:
            return None, f"API returned status code {response.status_code}"
        
        # XML namespaces for arXiv API
        ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        # Parse XML response
        root = ET.fromstring(response.content)
        entry = root.find('.//atom:entry', ns)
        if entry is None:
            return None, f"No paper found with ID {arxiv_id}"
        
        # Extract all details
        title_elem = entry.find('./atom:title', ns)
        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
        
        summary_elem = entry.find('./atom:summary', ns)
        summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None and summary_elem.text else ""
        
        # Get dates
        published_elem = entry.find('./atom:published', ns)
        published_date = ""
        if published_elem is not None and published_elem.text:
            try:
                published_date = datetime.strptime(published_elem.text, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
            except ValueError:
                published_date = ""
        
        updated_elem = entry.find('./atom:updated', ns)
        updated_date = ""
        if updated_elem is not None and updated_elem.text:
            try:
                updated_date = datetime.strptime(updated_elem.text, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
            except ValueError:
                updated_date = ""
        
        # Extract authors
        authors = []
        author_elements = entry.findall('./atom:author', ns)
        for author_elem in author_elements:
            name_elem = author_elem.find('./atom:name', ns)
            if name_elem is not None and name_elem.text:
                authors.append(name_elem.text.strip())
        
        # Extract comment
        comment_elem = entry.find('./arxiv:comment', ns)
        comment = comment_elem.text.strip() if comment_elem is not None and comment_elem.text else ""
        
        # Extract PDF link
        pdf_link = ""
        link_elements = entry.findall('./atom:link', ns)
        for link_elem in link_elements:
            if link_elem.get('type') == 'application/pdf':
                pdf_link = link_elem.get('href', '')
                break
        
        # Extract primary category
        primary_category_elem = entry.find('./arxiv:primary_category', ns)
        primary_category = primary_category_elem.get('term', '') if primary_category_elem is not None else ""
        
        # Extract all categories
        all_categories = []
        category_elements = entry.findall('./atom:category', ns)
        for category_elem in category_elements:
            term = category_elem.get('term', '')
            if term:
                all_categories.append(term)
        
        # Extract version information
        version = ""
        id_elem = entry.find('./atom:id', ns)
        if id_elem is not None and id_elem.text:
            version_match = re.search(r'v(\d+)$', id_elem.text)
            if version_match:
                version = version_match.group(1)
        
        return {
            'title': title,
            'authors': authors,
            'all_categories': all_categories,
            'primary_category': primary_category,
            'description': summary,
            'comment': comment,
            'version': version,
            'date_published': published_date,
            'date_updated': updated_date,
            'pdf_link': pdf_link
        }, None
        
    except Exception as e:
        return None, f"Error fetching paper details: {str(e)}"


def auto_fill_missing_paper_info():
    """
    Auto-fill missing information for all papers in the database.
    This function should be called from within Papers.py where st.session_state.paper exists.
    """
    if 'paper' not in st.session_state or st.session_state.paper.empty:
        st.error("No papers found in the database!")
        return
    
    paper_df = st.session_state.paper.copy()
    updated_count = 0
    failed_count = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for index, paper in paper_df.iterrows():
        # Update progress
        progress = (index + 1) / len(paper_df)
        progress_bar.progress(progress)
        status_text.text(f"Processing paper {index + 1} of {len(paper_df)}: {paper['Title'][:50]}...")
        
        # Skip if not an ArXiv paper
        if not is_arxiv_url(paper.get('Link')):
            continue
        
        # Check if paper needs updating
        needs_update_result, missing_fields = needs_paper_update(paper)
        if not needs_update_result:
            continue
        
        # Fetch enhanced details
        enhanced_data, error = fetch_enhanced_paper_details(paper['Link'])
        if error:
            failed_count += 1
            continue
        
        # Update the paper with missing information
        updated_fields = update_paper_fields(paper_df, index, enhanced_data)
        if updated_fields:
            updated_count += 1
        
        # Small delay to be respectful to the API
        import time
        time.sleep(0.5)
    
    # Update the session state
    st.session_state.paper = paper_df
    save_paper(paper_df)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    st.success(f"✅ Auto-fill completed!")
    st.info(f"📊 Updated {updated_count} papers, {failed_count} failed")
    
    return updated_count, failed_count


def needs_paper_update(paper):
    """Check if a paper needs updating based on missing fields."""
    missing_fields = []
    
    # Check for missing fields
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
    
    if not paper.get('Date Published'):
        missing_fields.append('Date Published')
    
    return len(missing_fields) > 0, missing_fields


def update_paper_fields(paper_df, index, enhanced_data):
    """Update paper fields with enhanced data."""
    updated_fields = []
    
    # Update Authors
    if (not paper_df.at[index, 'Authors'] or 
        (isinstance(paper_df.at[index, 'Authors'], list) and len(paper_df.at[index, 'Authors']) == 0)):
        if enhanced_data['authors']:
            paper_df.at[index, 'Authors'] = enhanced_data['authors']
            updated_fields.append('Authors')
    
    # Update Date Updated
    if not paper_df.at[index, 'Date Updated'] and enhanced_data['date_updated']:
        paper_df.at[index, 'Date Updated'] = enhanced_data['date_updated']
        updated_fields.append('Date Updated')
    
    # Update Comment
    if not paper_df.at[index, 'Comment'] and enhanced_data['comment']:
        paper_df.at[index, 'Comment'] = enhanced_data['comment']
        updated_fields.append('Comment')
    
    # Update PDF Link
    if not paper_df.at[index, 'PDF Link'] and enhanced_data['pdf_link']:
        paper_df.at[index, 'PDF Link'] = enhanced_data['pdf_link']
        updated_fields.append('PDF Link')
    
    # Update Primary Category
    if not paper_df.at[index, 'Primary Category'] and enhanced_data['primary_category']:
        paper_df.at[index, 'Primary Category'] = enhanced_data['primary_category']
        updated_fields.append('Primary Category')
    
    # Update All Categories
    if (not paper_df.at[index, 'All Categories'] or 
        (isinstance(paper_df.at[index, 'All Categories'], list) and len(paper_df.at[index, 'All Categories']) == 0)):
        if enhanced_data['all_categories']:
            paper_df.at[index, 'All Categories'] = enhanced_data['all_categories']
            updated_fields.append('All Categories')
    
    # Update Version
    if not paper_df.at[index, 'Version'] and enhanced_data['version']:
        paper_df.at[index, 'Version'] = enhanced_data['version']
        updated_fields.append('Version')
    
    # Update Date Published
    if not paper_df.at[index, 'Date Published'] and enhanced_data['date_published']:
        paper_df.at[index, 'Date Published'] = enhanced_data['date_published']
        updated_fields.append('Date Published')
    
    return updated_fields


def is_arxiv_url(url):
    """Check if URL is from ArXiv."""
    if not url:
        return False
    return 'arxiv.org' in str(url).lower()