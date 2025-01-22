# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # AI Safety Papers Visualization - Phase 1
#
# This notebook implements Phase 1 of the AI Safety Papers visualization project:
# 1. Metadata Collection using arXiv OAI-PMH API
# 2. Abstract Embedding Generation using ModernBERT-large
# 3. Initial Clustering using UMAP and HDBSCAN

# %%
# Clone repository if running in Colab
import os
if 'COLAB_GPU' in os.environ:
    # Clone the repository
    !git clone https://github.com/moiri-gamboni/ai-safety-landscape.git
    %cd ai-safety-landscape

# Install required packages
%pip install -r requirements.txt

# %% [markdown]
# ## Load Existing Database

# %%
import sqlite3
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check if database exists in Drive
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"
if os.path.exists(db_path):
    print(f"Found existing database at {db_path}")
    !cp "{db_path}" papers.db
    
    # Print existing data summary
    conn = sqlite3.connect('papers.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM papers')
    print(f"Database contains {c.fetchone()[0]} papers")
else:
    print("No existing database found in Drive. Will create new one.")


# %% [markdown]
# ## Create Database

# %%
import sqlite3
import os

def create_database():
    """Create SQLite database with necessary tables"""
    conn = sqlite3.connect('papers.db')
    c = conn.cursor()
    
    # Create papers table with all arXiv metadata fields
    c.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,           -- Required, maxOccurs=1
            title TEXT,                    -- Optional, maxOccurs=1
            abstract TEXT,                 -- Optional, maxOccurs=1
            categories TEXT,               -- Optional, maxOccurs=1
            msc_class TEXT,                -- Optional, maxOccurs=1
            acm_class TEXT,                -- Optional, maxOccurs=1
            doi TEXT,                      -- Optional, maxOccurs=1
            license TEXT,                  -- Optional, maxOccurs=1
            comments TEXT,                 -- Optional, maxOccurs=1
            created TEXT,                  -- Date of first version
            updated TEXT,                  -- Date of latest version
            withdrawn BOOLEAN DEFAULT 0,    -- Indicates if paper is withdrawn
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create paper versions table to track version history
    c.execute('''
        CREATE TABLE IF NOT EXISTS paper_versions (
            paper_id TEXT,
            version INTEGER,
            source_type TEXT,              -- D for Document, I for Inactive
            size TEXT,                     -- Size in kb
            date TEXT,                     -- Submission date of this version
            PRIMARY KEY (paper_id, version),
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
    ''')
    
    # Create authors table matching arXiv schema
    c.execute('''
        CREATE TABLE IF NOT EXISTS authors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyname TEXT NOT NULL,          -- Required, maxOccurs=1
            forenames TEXT,                 -- Optional, maxOccurs=1
            suffix TEXT,                    -- Optional, maxOccurs=1
            UNIQUE(keyname, forenames, suffix)  -- Avoid duplicate authors
        )
    ''')
    
    # Create author affiliations table (since it's maxOccurs=unbounded in schema)
    c.execute('''
        CREATE TABLE IF NOT EXISTS author_affiliations (
            author_id INTEGER,
            affiliation TEXT NOT NULL,
            PRIMARY KEY (author_id, affiliation),
            FOREIGN KEY (author_id) REFERENCES authors(id)
        )
    ''')
    
    # Create paper_authors junction table
    c.execute('''
        CREATE TABLE IF NOT EXISTS paper_authors (
            paper_id TEXT,
            author_id INTEGER,
            author_position INTEGER,        -- Track author order in paper
            PRIMARY KEY (paper_id, author_id),
            FOREIGN KEY (paper_id) REFERENCES papers(id),
            FOREIGN KEY (author_id) REFERENCES authors(id)
        )
    ''')
    
    # Create indices for common queries after all tables are created
    c.execute("CREATE INDEX IF NOT EXISTS idx_categories ON papers(categories)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_withdrawn ON papers(withdrawn)")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_authors_unique ON authors(keyname, forenames, suffix)")
    
    conn.commit()
    return conn

# Initialize database
conn = create_database()

# %% [markdown]
# ## Harvest Metadata
#
# ### Configuration
# - **Number of Papers**: Control how many papers to fetch. Set to 0 to fetch all CS papers (warning: this will take a long time).
# - **Resumption Token**: If harvesting was interrupted, paste the resumption token here to continue from where you left off.
#   You can paste the full URL or just the token - URL-encoded characters (like %7C) will be automatically converted.

# %%
# @title Harvesting Configuration  {"run":"auto"}
num_papers = 100 # @param {type:"slider", min:0, max:10000, step:100}
resumption_token = "" # @param {type:"string"}

# %%
from sickle import Sickle
from sickle.models import Record
from tqdm import tqdm
import urllib.parse
import re

class ArxivRecord(Record):
    """Custom record class for arXiv metadata format"""
    def get_metadata(self):
        # Get the arXiv metadata namespace
        ns = {'arxiv': 'http://arxiv.org/OAI/arXiv/'}
        
        # Get the arXiv element which contains all metadata
        arxiv = self.xml.find('.//{http://arxiv.org/OAI/arXiv/}arXiv')
        if arxiv is None:
            raise ValueError("Could not find arXiv metadata element")
            
        metadata = {}
        
        # Required field - use identifier from header
        metadata['id'] = self.header.identifier
        if metadata['id'].startswith('oai:arXiv.org:'):
            metadata['id'] = metadata['id'].replace('oai:arXiv.org:', '')
        
        # Map arXiv metadata fields according to schema
        field_mapping = {
            'created': 'created',
            'updated': 'updated',
            'title': 'title',
            'abstract': 'abstract',
            'categories': 'categories',
            'msc_class': 'msc-class',
            'acm_class': 'acm-class',
            'report_no': 'report-no',
            'journal_ref': 'journal-ref',
            'doi': 'doi',
            'comments': 'comments',
            'license': 'license'
        }
        
        for field, xml_field in field_mapping.items():
            elem = arxiv.find(f'arxiv:{xml_field}', namespaces=ns)
            metadata[field] = elem.text if elem is not None and elem.text else None
        
        # Extract authors according to schema
        authors = []
        authors_elem = arxiv.find('arxiv:authors', namespaces=ns)
        if authors_elem is not None:
            for author_elem in authors_elem.findall('arxiv:author', namespaces=ns):
                if author_elem is None:
                    continue
                    
                author = {}
                
                # Required keyname field
                keyname_elem = author_elem.find('arxiv:keyname', namespaces=ns)
                if keyname_elem is None or not keyname_elem.text:
                    continue  # Skip authors without keyname
                author['keyname'] = keyname_elem.text.rstrip(',')  # Remove trailing comma if present
                
                # Optional author fields
                forenames_elem = author_elem.find('arxiv:forenames', namespaces=ns)
                author['forenames'] = forenames_elem.text if forenames_elem is not None and forenames_elem.text else None
                
                suffix_elem = author_elem.find('arxiv:suffix', namespaces=ns)
                author['suffix'] = suffix_elem.text if suffix_elem is not None and suffix_elem.text else None
                
                # Multiple affiliations possible
                author['affiliations'] = []
                for affiliation_elem in author_elem.findall('arxiv:affiliation', namespaces=ns):
                    if affiliation_elem is not None and affiliation_elem.text:
                        author['affiliations'].append(affiliation_elem.text)
                
                authors.append(author)
        
        metadata['authors'] = authors
        return metadata

class ArxivRawRecord(Record):
    """Custom record class for arXivRaw metadata format"""
    def get_metadata(self):
        # Get the arXivRaw metadata namespace
        ns = {'arxiv': 'http://arxiv.org/OAI/arXivRaw/'}
        
        # Get the arXivRaw element which contains all metadata
        arxiv = self.xml.find('.//{http://arxiv.org/OAI/arXivRaw/}arXivRaw')
        if arxiv is None:
            raise ValueError("Could not find arXivRaw metadata element")
            
        metadata = {}
        
        # Required field - use identifier from header
        metadata['id'] = self.header.identifier
        if metadata['id'].startswith('oai:arXiv.org:'):
            metadata['id'] = metadata['id'].replace('oai:arXiv.org:', '')
        
        # Get all versions and sort by version number
        versions = []
        for version_elem in arxiv.findall('.//arxiv:version', namespaces=ns):
            version_num = version_elem.get('version', 'v1').lstrip('v')
            try:
                version_num = int(version_num)
            except ValueError:
                continue
                
            version_info = {
                'version': version_num,
                'date': version_elem.find('arxiv:date', namespaces=ns).text if version_elem.find('arxiv:date', namespaces=ns) is not None else None,
                'size': version_elem.find('arxiv:size', namespaces=ns).text if version_elem.find('arxiv:size', namespaces=ns) is not None else None,
                'source_type': version_elem.find('arxiv:source_type', namespaces=ns).text if version_elem.find('arxiv:source_type', namespaces=ns) is not None else 'D'
            }
            versions.append(version_info)
        
        # Sort versions by version number
        versions.sort(key=lambda x: x['version'])
        metadata['versions'] = versions
        
        return metadata

def save_papers(papers, conn):
    """Save papers and authors to SQLite database"""
    c = conn.cursor()
    
    def normalize_suffix(suffix):
        """Normalize author suffixes to a standard format"""
        if not suffix:
            return None
        suffix = suffix.strip()
        # Normalize Jr variations
        if suffix.upper() in ['JR', 'JR.', 'JR ', 'JUNIOR']:
            return 'Jr.'
        # Normalize Sr variations
        if suffix.upper() in ['SR', 'SR.', 'SR ', 'SENIOR']:
            return 'Sr.'
        # Normalize roman numerals
        if suffix.upper() in ['I', 'II', 'III', 'IV', 'V']:
            return suffix.upper()
        return suffix
    
    # Prepare batch data
    paper_data = []
    author_data = set()  # Use set to deduplicate authors
    paper_author_data = []
    affiliation_data = set()  # Use set to deduplicate affiliations
    
    # Collect all data first
    for paper in papers:
        # Collect paper data
        paper_data.append((
            paper['id'],
            paper.get('title'),
            paper.get('abstract'),
            paper.get('categories'),
            paper.get('msc_class'),
            paper.get('acm_class'),
            paper.get('doi'),
            paper.get('license'),
            paper.get('comments')
        ))
        
        # Collect author and affiliation data
        for pos, author in enumerate(paper['authors'], 1):
            # Normalize suffix
            suffix = normalize_suffix(author.get('suffix'))
            
            # Add author to set
            author_tuple = (
                author['keyname'],
                author.get('forenames'),
                suffix
            )
            author_data.add(author_tuple)
            
            # Store paper-author relationship with position
            paper_author_data.append((
                paper['id'],
                author_tuple,  # We'll replace this with ID after inserting authors
                pos
            ))
            
            # Add affiliations to set
            for affiliation in author.get('affiliations', []):
                affiliation_data.add((author_tuple, affiliation))
    
    try:
        # Batch insert papers
        c.executemany('''
            INSERT OR IGNORE INTO papers (
                id, title, abstract, categories,
                msc_class, acm_class, doi, license, comments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', paper_data)
        
        # Batch insert authors and get their IDs
        author_id_map = {}  # Map author tuple to ID
        c.executemany('''
            INSERT OR IGNORE INTO authors (keyname, forenames, suffix)
            VALUES (?, ?, ?)
        ''', author_data)
        
        # Get all author IDs in one query
        placeholders = ','.join(['(?, ?, ?)'] * len(author_data))
        c.execute(f'''
            SELECT id, keyname, forenames, suffix 
            FROM authors 
            WHERE (keyname, forenames, suffix) IN ({placeholders})
        ''', [val for author in author_data for val in author])
        
        for row in c.fetchall():
            author_id_map[author_tuple] = row[0]
        
        # Update paper-author data with real IDs
        paper_author_final = []
        for paper_id, author_tuple, pos in paper_author_data:
            try:
                author_id = author_id_map[author_tuple]
                paper_author_final.append((paper_id, author_id, pos))
            except KeyError:
                print(f"Warning: Could not find ID for author {author_tuple}")
        
        # Batch insert paper-author relationships
        c.executemany('''
            INSERT OR IGNORE INTO paper_authors (paper_id, author_id, author_position)
            VALUES (?, ?, ?)
        ''', paper_author_final)
        
        # Batch insert affiliations
        affiliation_final = [
            (author_id_map[author_tuple], affiliation)
            for author_tuple, affiliation in affiliation_data
        ]
        c.executemany('''
            INSERT OR IGNORE INTO author_affiliations (author_id, affiliation)
            VALUES (?, ?)
        ''', affiliation_final)
        
        conn.commit()
        
    except sqlite3.IntegrityError as e:
        print(f"Error during batch operations: {str(e)}")
        conn.rollback()

def save_versions(paper_versions, conn):
    """Save paper version information to database efficiently"""
    c = conn.cursor()
    
    # Prepare batch data
    paper_updates = []
    version_inserts = []
    
    # Collect all data first
    for paper_id, versions in paper_versions.items():
        # Sort versions by version number
        versions = sorted(versions, key=lambda x: x['version'])
        
        if versions:
            # Get first and latest version dates
            created = versions[0]['date']
            updated = versions[-1]['date']
            
            # Check if paper is withdrawn based on latest version
            latest_version = versions[-1]
            withdrawn = (
                latest_version['source_type'] == 'I' or
                latest_version['size'] == '0kb'
            )
            
            # Add to paper updates
            paper_updates.append((created, updated, withdrawn, paper_id))
            
            # Add all versions
            for version in versions:
                version_inserts.append((
                    paper_id,
                    version['version'],
                    version['source_type'],
                    version['size'],
                    version['date']
                ))
    
    try:
        # Batch update papers
        c.executemany('''
            UPDATE papers 
            SET created = ?, updated = ?, withdrawn = ?
            WHERE id = ?
        ''', paper_updates)
        
        # Batch insert versions
        c.executemany('''
            INSERT OR IGNORE INTO paper_versions (
                paper_id, version, source_type, size, date
            ) VALUES (?, ?, ?, ?, ?)
        ''', version_inserts)
        
        conn.commit()
        
    except sqlite3.IntegrityError as e:
        print(f"Error during batch operations: {str(e)}")
        conn.rollback()

def fetch_arxiv_records(metadata_prefix, record_class, max_results=None, resumption_token=None):
    """Base function for fetching records from arXiv OAI-PMH API"""
    sickle = Sickle('http://export.arxiv.org/oai2',
                    max_retries=5,
                    retry_status_codes=[503],
                    default_retry_after=20)
    
    # Register our custom record class
    sickle.class_mapping['ListRecords'] = record_class
    
    try:
        # Use ListRecords with specified format
        if resumption_token:
            # Extract token from URL if full URL was pasted
            if 'resumptionToken=' in resumption_token:
                resumption_token = re.search(r'resumptionToken=([^&]+)', resumption_token).group(1)
            # Unescape the token
            resumption_token = urllib.parse.unquote(resumption_token)
            records = sickle.ListRecords(resumptionToken=resumption_token)
        else:
            records = sickle.ListRecords(
                metadataPrefix=metadata_prefix,
                set='cs',
                ignore_deleted=True
            )
        
        # Process records with progress bar
        results = []
        with tqdm(desc=f"Fetching {metadata_prefix}", unit=" papers") as pbar:
            while True:
                try:
                    record = next(records)
                    try:
                        metadata = record.get_metadata()
                        if metadata and metadata.get('id'):  # Only add if we have valid metadata
                            results.append(metadata)
                            pbar.update(1)
                            
                            # Respect max_results limit if set
                            if max_results and len(results) >= max_results:
                                break
                    except Exception as e:
                        print(f"Error processing record: {str(e)}")
                        continue
                except StopIteration:
                    break
                except Exception as e:
                    # Get the current resumption token from the Sickle iterator
                    if hasattr(records, 'resumption_token'):
                        print(f"\nError occurred. Current resumption token: {records.resumption_token}")
                    raise  # Re-raise the exception to let Sickle's retry mechanism handle it
                    
    except Exception as e:
        print(f"Error during harvesting: {str(e)}")
        # Print final resumption token if available
        if hasattr(records, 'resumption_token'):
            print(f"Final resumption token: {records.resumption_token}")
        
    return results

def fetch_cs_papers(max_results=None, resumption_token=None):
    """Fetch CS papers from arXiv using OAI-PMH with arXiv metadata format"""
    return fetch_arxiv_records('arXiv', ArxivRecord, max_results, resumption_token)

def fetch_raw_metadata(max_results=None, resumption_token=None):
    """Fetch paper version information from arXiv using OAI-PMH with arXivRaw metadata format"""
    return fetch_arxiv_records('arXivRaw', ArxivRawRecord, max_results, resumption_token)

# %% [markdown]
# ## Harvest arXiv Metadata
# This cell fetches the main metadata (titles, abstracts, authors, etc.) from arXiv's OAI-PMH API.
# You can run this cell independently and save the database before proceeding to raw metadata harvesting.

# %%
# Fetch papers using the slider value (None for all papers)
# First check current database count
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM papers')
current_count = c.fetchone()[0]
print(f"Current papers in database before harvesting: {current_count}")

# Fetch papers with arXiv metadata
initial_papers = fetch_cs_papers(
    max_results=num_papers if num_papers > 0 else None,
    resumption_token=resumption_token if resumption_token else None
)
save_papers(initial_papers, conn)

# Print final count
c.execute('SELECT COUNT(*) FROM papers')
final_count = c.fetchone()[0]
print(f"\nMetadata harvesting complete:")
print(f"- Papers added this session: {final_count - current_count}")
print(f"- Total papers in database: {final_count}")

# %% [markdown]
# ## Harvest arXiv Raw Metadata
# This cell fetches additional metadata about paper versions from arXiv's OAI-PMH API using the arXivRaw format.
# You can run this cell separately after running the main metadata harvesting above.
# 
# **Note**: Make sure to run the database loading cell first if you're running this in a new session.

# %%
# Fetch version information with arXivRaw
paper_versions = fetch_raw_metadata(
    max_results=num_papers if num_papers > 0 else None,
    resumption_token=resumption_token if resumption_token else None
)
save_versions(paper_versions, conn)

# Print final stats about versions
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM paper_versions')
version_count = c.fetchone()[0]
c.execute('SELECT COUNT(DISTINCT paper_id) FROM paper_versions')
papers_with_versions = c.fetchone()[0]
print(f"\nRaw metadata harvesting complete:")
print(f"- Total versions stored: {version_count}")
print(f"- Papers with version info: {papers_with_versions}")

# %% [markdown]
# ## Data Quality Check

# %%
def inspect_papers(conn, limit=5):
    """Print detailed information for a sample of papers"""
    c = conn.cursor()
    
    # Get sample papers with their authors
    c.execute('''
        SELECT p.id, p.title, p.abstract, p.categories, p.created, p.updated,
               GROUP_CONCAT(a.keyname || COALESCE(', ' || a.forenames, '') || COALESCE(' ' || a.suffix, ''))
        FROM papers p
        LEFT JOIN paper_authors pa ON p.id = pa.paper_id
        LEFT JOIN authors a ON pa.author_id = a.id
        GROUP BY p.id
        LIMIT ?
    ''', (limit,))
    
    papers = c.fetchall()
    
    print(f"Inspecting {len(papers)} sample papers:\n")
    for paper in papers:
        print("=" * 80)
        print(f"ID: {paper[0]}")
        print(f"Title: {paper[1]}")
        print(f"Abstract: {paper[2][:200]}..." if paper[2] else "Abstract: None")
        print(f"Categories: {paper[3]}")
        print(f"Created: {paper[4]}")
        print(f"Updated: {paper[5]}")
        print(f"Authors: {paper[6]}")
        print()
    
    # Print detailed statistics
    print("\nDatabase Statistics:")
    
    # Paper statistics
    c.execute('SELECT COUNT(*) FROM papers')
    total_papers = c.fetchone()[0]
    print(f"\nPapers:")
    print(f"Total Papers: {total_papers}")
    
    # Count cs.AI papers
    c.execute("SELECT COUNT(*) FROM papers WHERE categories LIKE '%cs.AI%'")
    ai_papers = c.fetchone()[0]
    print(f"CS.AI Papers: {ai_papers} ({(ai_papers/total_papers*100):.1f}% of total)")
    
    # Category statistics
    print("\nTop Categories:")
    c.execute('''
        SELECT category, COUNT(*) as count
        FROM (
            SELECT TRIM(value) as category
            FROM papers, json_each('["' || REPLACE(categories, ' ', '","') || '"]')
            WHERE categories IS NOT NULL
        )
        WHERE category != ''
        GROUP BY category
        ORDER BY count DESC
        LIMIT 10
    ''')
    for category, count in c.fetchall():
        if category:  # Skip empty category
            percentage = (count / total_papers) * 100
            print(f"{category}: {count} papers ({percentage:.1f}%)")
    
    # Author statistics
    print(f"\nAuthors:")
    c.execute('SELECT COUNT(*) FROM authors')
    total_authors = c.fetchone()[0]
    print(f"Total Authors: {total_authors}")
    
    c.execute('SELECT COUNT(*) FROM paper_authors')
    total_author_links = c.fetchone()[0]
    print(f"Total Author-Paper Links: {total_author_links}")
    
    # Papers per author distribution
    print("\nPapers per author distribution:")
    c.execute('''
        SELECT papers_count, COUNT(*) as authors_with_this_many_papers
        FROM (
            SELECT author_id, COUNT(*) as papers_count
            FROM paper_authors
            GROUP BY author_id
        )
        GROUP BY papers_count
        ORDER BY papers_count
        LIMIT 10
    ''')
    for paper_count, author_count in c.fetchall():
        print(f"{author_count} authors have {paper_count} paper(s)")
    
    c.execute('''
        SELECT COUNT(*) FROM 
        (SELECT author_id FROM paper_authors GROUP BY author_id HAVING COUNT(*) > 1)
    ''')
    authors_multiple_papers = c.fetchone()[0]
    print(f"\nAuthors with Multiple Papers: {authors_multiple_papers}")
    
    c.execute('SELECT AVG(author_count) FROM (SELECT paper_id, COUNT(*) as author_count FROM paper_authors GROUP BY paper_id)')
    avg_authors_per_paper = c.fetchone()[0]
    print(f"\nAverage Authors per Paper: {avg_authors_per_paper:.2f}")
    
    # Affiliation statistics
    print("\nAffiliations:")
    c.execute('SELECT COUNT(DISTINCT affiliation) FROM author_affiliations')
    total_unique_affiliations = c.fetchone()[0]
    print(f"Unique Affiliations: {total_unique_affiliations}")
    
    c.execute('SELECT COUNT(*) FROM author_affiliations')
    total_affiliation_links = c.fetchone()[0]
    print(f"Total Author-Affiliation Links: {total_affiliation_links}")
    
    c.execute('''
        SELECT COUNT(*) FROM 
        (SELECT author_id FROM author_affiliations GROUP BY author_id)
    ''')
    authors_with_affiliations = c.fetchone()[0]
    print(f"Authors with Affiliations: {authors_with_affiliations} ({(authors_with_affiliations/total_authors*100):.1f}% of authors)")
    
    print("\nTop 10 Affiliations:")
    c.execute('''
        SELECT affiliation, COUNT(*) as count
        FROM author_affiliations
        GROUP BY affiliation
        ORDER BY count DESC
        LIMIT 10
    ''')
    for affiliation, count in c.fetchall():
        print(f"- {affiliation}: {count} authors")
    
    # Metadata field statistics
    print("\nMetadata Field Coverage:")
    fields = [
        'title', 'abstract', 'categories', 'created', 'updated',
        'msc_class', 'acm_class', 'doi', 'license'
    ]
    
    for field in fields:
        c.execute(f'SELECT COUNT(*) FROM papers WHERE {field} IS NOT NULL')
        present = c.fetchone()[0]
        percentage = (present / total_papers) * 100 if total_papers > 0 else 0
        print(f"{field}: {present}/{total_papers} ({percentage:.1f}%)")

    # Add withdrawn paper statistics after paper statistics
    print("\nWithdrawn Paper Statistics:")
    c.execute('SELECT COUNT(*) FROM papers WHERE withdrawn = 1')
    withdrawn_count = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM papers WHERE withdrawn = 0')
    active_count = c.fetchone()[0]
    print(f"Active Papers: {active_count} ({(active_count/total_papers*100):.1f}% of total)")
    print(f"Withdrawn Papers: {withdrawn_count} ({(withdrawn_count/total_papers*100):.1f}% of total)")
    
    # Check for papers with "withdrawn" in comments but not marked withdrawn
    c.execute('''
        SELECT id, title, comments 
        FROM papers 
        WHERE comments LIKE '%withdrawn%' AND withdrawn = 0
    ''')
    potential_withdrawn = c.fetchall()
    if potential_withdrawn:
        print(f"\nFound {len(potential_withdrawn)} papers with 'withdrawn' in comments but not marked withdrawn:")
        for i, paper in enumerate(potential_withdrawn[:5], 1):  # Show up to 5 examples
            print(f"\n{i}. Paper ID: {paper[0]}")
            print(f"   Title: {paper[1]}")
            print(f"   Comments: {paper[2]}")
    else:
        print("\nNo papers found with 'withdrawn' in comments but not marked withdrawn")

# Inspect sample papers
inspect_papers(conn) 

# %% [markdown]
# ## Save Database to Drive

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create project directory if it doesn't exist
!mkdir -p "/content/drive/MyDrive/ai-safety-papers"

# Copy database to Drive
!cp papers.db "/content/drive/MyDrive/ai-safety-papers/papers.db"
print("Database saved to Google Drive at: /ai-safety-papers/papers.db") 