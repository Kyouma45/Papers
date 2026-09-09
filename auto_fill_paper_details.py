#!/usr/bin/env python3
"""
Auto-fill Script for Paper Details

This script automatically extracts and fills missing information for existing papers
in the paper_with_topics.json file using the ArXiv API. It will update entries with:

- Authors - Multiple authors with their names
- Updated date - When the paper was last updated on ArXiv
- Comment - Additional metadata about the paper (pages, appendix info, etc.)
- PDF link - Direct link to the PDF version
- Primary category - The main subject classification
- Version information - From the ID (v2, v3, etc.)
- All categories - Not just primary categories, but all subject classifications
- Proper date formatting - Ensures dates are consistently formatted

Usage:
    python auto_fill_paper_details.py [--dry-run] [--backup] [--verbose]

Arguments:
    --dry-run    : Show what would be updated without making changes
    --backup     : Create a backup of the original file
    --verbose    : Show detailed progress information
    --help       : Show this help message
"""

import json
import argparse
import shutil
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import urllib3

# Suppress SSL warning messages
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PAPER_FILE = "paper_with_topics.json"
BACKUP_SUFFIX = ".backup"


def extract_arxiv_id(url):
    """Extract arXiv ID from URL or return the ID if directly provided."""
    if not url:
        return None
    
    # Clean up the URL
    url = str(url).strip()
    
    # Match patterns like arxiv.org/abs/2312.12345 or arxiv.org/pdf/2312.12345.pdf
    patterns = [
        r'arxiv\.org/(?:abs|pdf)/(\d+\.\d+)',
        r'(\d{4}\.\d{5,6})'  # Direct arXiv ID format
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_paper_details_from_arxiv(arxiv_url, verbose=False):
    """
    Fetch paper details from ArXiv API.
    
    Returns:
        tuple: (paper_details_dict, error_message)
               paper_details_dict is None if there was an error
    """
    try:
        # Extract arXiv ID from URL
        arxiv_id = extract_arxiv_id(arxiv_url)
        if not arxiv_id:
            return None, f"Invalid arXiv URL format: {arxiv_url}"
        
        if verbose:
            print(f"📡 Fetching details for arXiv ID: {arxiv_id}")
        
        # Make request to arXiv API
        api_url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        
        try:
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
            
            # Check if paper exists
            entry = root.find('.//atom:entry', ns)
            if entry is None:
                return None, f"No paper found with ID {arxiv_id}"
            
            # Extract paper details
            title_elem = entry.find('./atom:title', ns)
            title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
            
            summary_elem = entry.find('./atom:summary', ns)
            summary = summary_elem.text.strip().replace('\n', ' ') if summary_elem is not None and summary_elem.text else ""
            
            # Get publication and update dates
            published_elem = entry.find('./atom:published', ns)
            published = published_elem.text if published_elem is not None else None
            published_date = ""
            if published:
                try:
                    published_date = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
                except ValueError:
                    published_date = ""
            
            updated_elem = entry.find('./atom:updated', ns)
            updated = updated_elem.text if updated_elem is not None else None
            updated_date = ""
            if updated:
                try:
                    updated_date = datetime.strptime(updated, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d')
                except ValueError:
                    updated_date = ""
            
            # Extract authors
            authors = []
            author_elements = entry.findall('./atom:author', ns)
            for author_elem in author_elements:
                name_elem = author_elem.find('./atom:name', ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text.strip())
            
            # Extract comment (additional metadata)
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
            
            # Extract version information from ID
            version = ""
            id_elem = entry.find('./atom:id', ns)
            if id_elem is not None and id_elem.text:
                id_text = id_elem.text
                version_match = re.search(r'v(\d+)$', id_text)
                if version_match:
                    version = version_match.group(1)
            
            # Create paper details dictionary
            paper_details = {
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
            }
            
            if verbose:
                print(f"✅ Successfully extracted details for: {title[:60]}{'...' if len(title) > 60 else ''}")
            
            return paper_details, None
            
        except requests.exceptions.RequestException as e:
            return None, f"Request error: {str(e)}"
            
    except Exception as e:
        return None, f"Error fetching paper details: {str(e)}"


def load_papers_data(file_path):
    """Load papers data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File '{file_path}' not found!")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in '{file_path}': {e}")
        return None


def save_papers_data(papers, file_path):
    """Save papers data to JSON file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"ERROR saving to '{file_path}': {e}")
        return False


def create_backup(file_path):
    """Create a backup of the original file."""
    backup_path = file_path + BACKUP_SUFFIX
    try:
        shutil.copy2(file_path, backup_path)
        print(f"Backup created: {backup_path}")
        return True
    except Exception as e:
        print(f"Error creating backup: {e}")
        return False


def is_arxiv_url(url):
    """Check if URL is from ArXiv."""
    if not url:
        return False
    return 'arxiv.org' in str(url).lower()


def needs_update(paper):
    """
    Check if a paper entry needs updating based on missing fields.
    
    Returns:
        tuple: (needs_update: bool, missing_fields: list)
    """
    missing_fields = []
    
    # Check for missing or empty fields that we can extract from ArXiv
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
    
    # Check if Date Published is missing (but only for ArXiv papers)
    if not paper.get('Date Published') and is_arxiv_url(paper.get('Link')):
        missing_fields.append('Date Published')
    
    return len(missing_fields) > 0, missing_fields


def update_paper_with_arxiv_data(paper, arxiv_data):
    """
    Update paper entry with data from ArXiv API.
    
    Returns:
        tuple: (updated_paper, fields_updated)
    """
    updated_paper = paper.copy()
    fields_updated = []
    
    # Update Authors (if missing or empty)
    if (not paper.get('Authors') or 
        (isinstance(paper.get('Authors'), list) and len(paper.get('Authors')) == 0)) and arxiv_data['authors']:
        updated_paper['Authors'] = arxiv_data['authors']
        fields_updated.append('Authors')
    
    # Update Date Updated
    if not paper.get('Date Updated') and arxiv_data['date_updated']:
        updated_paper['Date Updated'] = arxiv_data['date_updated']
        fields_updated.append('Date Updated')
    
    # Update Comment
    if not paper.get('Comment') and arxiv_data['comment']:
        updated_paper['Comment'] = arxiv_data['comment']
        fields_updated.append('Comment')
    
    # Update PDF Link
    if not paper.get('PDF Link') and arxiv_data['pdf_link']:
        updated_paper['PDF Link'] = arxiv_data['pdf_link']
        fields_updated.append('PDF Link')
    
    # Update Primary Category
    if not paper.get('Primary Category') and arxiv_data['primary_category']:
        updated_paper['Primary Category'] = arxiv_data['primary_category']
        fields_updated.append('Primary Category')
    
    # Update All Categories
    if (not paper.get('All Categories') or 
        (isinstance(paper.get('All Categories'), list) and len(paper.get('All Categories')) == 0)) and arxiv_data['all_categories']:
        updated_paper['All Categories'] = arxiv_data['all_categories']
        fields_updated.append('All Categories')
    
    # Update Version
    if not paper.get('Version') and arxiv_data['version']:
        updated_paper['Version'] = arxiv_data['version']
        fields_updated.append('Version')
    
    # Update Date Published (if missing and this is an ArXiv paper)
    if not paper.get('Date Published') and arxiv_data['date_published']:
        updated_paper['Date Published'] = arxiv_data['date_published']
        fields_updated.append('Date Published')
    
    # Update Description if the current one is empty or very short and we have a better one
    if (not paper.get('Description') or len(paper.get('Description', '')) < 10) and arxiv_data['description']:
        if len(arxiv_data['description']) > len(paper.get('Description', '')):
            updated_paper['Description'] = arxiv_data['description']
            fields_updated.append('Description')
    
    return updated_paper, fields_updated


def print_paper_summary(paper, index, total):
    """Print a summary of the paper being processed."""
    title = paper.get('Title', 'Unknown Title')
    link = paper.get('Link', 'No Link')
    print(f"\n[{index + 1}/{total}] {title[:60]}{'...' if len(title) > 60 else ''}")
    print(f"Link: {link}")


def print_update_summary(fields_updated, arxiv_data):
    """Print summary of what fields were updated."""
    if not fields_updated:
        print("   No updates needed - all fields already populated")
        return
    
    print(f"   Updated {len(fields_updated)} field(s): {', '.join(fields_updated)}")
    
    # Show some key details
    if 'Authors' in fields_updated and arxiv_data['authors']:
        authors_str = ', '.join(arxiv_data['authors'][:3])
        if len(arxiv_data['authors']) > 3:
            authors_str += f' (and {len(arxiv_data["authors"]) - 3} more)'
        print(f"      Authors: {authors_str}")
    
    if 'Primary Category' in fields_updated and arxiv_data['primary_category']:
        print(f"      Primary Category: {arxiv_data['primary_category']}")
    
    if 'All Categories' in fields_updated and arxiv_data['all_categories']:
        cats_str = ', '.join(arxiv_data['all_categories'][:3])
        if len(arxiv_data['all_categories']) > 3:
            cats_str += f' (and {len(arxiv_data["all_categories"]) - 3} more)'
        print(f"      All Categories: {cats_str}")
    
    if 'Version' in fields_updated and arxiv_data['version']:
        print(f"      Version: v{arxiv_data['version']}")
    
    if 'Date Updated' in fields_updated and arxiv_data['date_updated']:
        print(f"      Last Updated: {arxiv_data['date_updated']}")


def main():
    # Set UTF-8 encoding for Windows compatibility
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    parser = argparse.ArgumentParser(description="Auto-fill missing paper details from ArXiv API")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be updated without making changes')
    parser.add_argument('--backup', action='store_true', 
                       help='Create a backup of the original file')
    parser.add_argument('--verbose', action='store_true', 
                       help='Show detailed progress information')
    parser.add_argument('--file', default=PAPER_FILE, 
                       help=f'Path to the papers JSON file (default: {PAPER_FILE})')
    
    args = parser.parse_args()
    
    print("ArXiv Paper Details Auto-Fill Script")
    print("=" * 50)
    
    # Check if file exists
    if not Path(args.file).exists():
        print(f"ERROR: File '{args.file}' not found!")
        return 1
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        if not create_backup(args.file):
            return 1
    
    # Load papers data
    print(f"Loading papers from: {args.file}")
    papers = load_papers_data(args.file)
    if papers is None:
        return 1
    
    print(f"Found {len(papers)} papers total")
    
    # Filter papers that need updating and are from ArXiv
    papers_to_update = []
    for i, paper in enumerate(papers):
        if is_arxiv_url(paper.get('Link')):
            needs_upd, missing_fields = needs_update(paper)
            if needs_upd:
                papers_to_update.append((i, paper, missing_fields))
    
    print(f"Found {len(papers_to_update)} ArXiv papers that need updating")
    
    if len(papers_to_update) == 0:
        print("All papers already have complete information!")
        return 0
    
    if args.dry_run:
        print("\nDRY RUN MODE - Showing what would be updated:")
        print("-" * 50)
    
    # Process each paper that needs updating
    updated_papers = papers.copy()
    total_updated = 0
    failed_updates = 0
    
    for paper_index, (original_index, paper, missing_fields) in enumerate(papers_to_update):
        print_paper_summary(paper, paper_index, len(papers_to_update))
        
        if args.verbose:
            print(f"   ❓ Missing fields: {', '.join(missing_fields)}")
        
        # Fetch details from ArXiv
        arxiv_data, error = fetch_paper_details_from_arxiv(paper.get('Link'), verbose=args.verbose)
        
        if error:
            print(f"   Failed to fetch details: {error}")
            failed_updates += 1
            continue
        
        # Update paper with ArXiv data
        updated_paper, fields_updated = update_paper_with_arxiv_data(paper, arxiv_data)
        
        if args.dry_run:
            print_update_summary(fields_updated, arxiv_data)
        else:
            # Actually update the paper
            updated_papers[original_index] = updated_paper
            print_update_summary(fields_updated, arxiv_data)
            
            if fields_updated:
                total_updated += 1
        
        # Add a small delay to be respectful to the API
        import time
        time.sleep(0.5)
    
    # Save updated data
    if not args.dry_run and total_updated > 0:
        print(f"\nSaving updated data to: {args.file}")
        if save_papers_data(updated_papers, args.file):
            print("Successfully saved updated papers data!")
        else:
            print("Failed to save updated papers data!")
            return 1
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("-" * 20)
    print(f"Total papers: {len(papers)}")
    print(f"Papers needing updates: {len(papers_to_update)}")
    
    if args.dry_run:
        print(f"Would update: {len(papers_to_update) - failed_updates} papers")
        print(f"Would fail: {failed_updates} papers")
        print("\nRun without --dry-run to apply changes")
    else:
        print(f"Successfully updated: {total_updated} papers")
        print(f"Failed to update: {failed_updates} papers")
        if args.backup:
            print(f"Backup saved as: {args.file}{BACKUP_SUFFIX}")
    
    return 0


if __name__ == "__main__":
    exit(main())