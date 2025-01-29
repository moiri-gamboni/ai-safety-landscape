# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # arXiv Computer Science Papers Database
#
# This notebook creates and maintains a SQLite database of Computer Science papers from arXiv. It:
# 1. Harvests metadata using arXiv's OAI-PMH API
# 2. Stores paper metadata, versions, and author information
# 3. Provides data quality analysis and management tools
#
# The database schema includes:
# - Papers: Core metadata (title, abstract, categories, etc.)
# - Paper Versions: Version history and submission dates
# - Authors: Normalized author information
# - Paper-Author relationships: Author order in papers

# %% [markdown]
# # 1. Database Setup and Initialization

# %% [markdown]
# ## 1.1 Environment Setup
# Run this cell to set up the environment if using Google Colab.

# %%
# Install required packages if running in Colab
import os
if 'COLAB_GPU' in os.environ:
    # Install only the packages needed for this notebook
    %pip install sickle tqdm # pyright: ignore

# %% [markdown]
# ## 1.2 Database Initialization
# Choose whether to load an existing database from Google Drive or create a new one.

# %%
# @title Database Initialization Choice
db_choice = "create_new" # @param ["create_new", "load_existing"] {type:"string", label:"Database Choice"}

import sqlite3
import os

def load_existing_database():
    """Load existing database from Google Drive"""
    # Mount Google Drive
    from google.colab import drive # pyright: ignore [reportMissingImports]
    drive.mount('/content/drive')

    # Check if database exists in Drive
    db_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"No database found at {db_path}")
        
    print(f"Found existing database at {db_path}")
    %cp "{db_path}" papers.db # pyright: ignore
    
    # Connect and print summary
    conn = sqlite3.connect('papers.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM papers')
    print(f"Database contains {c.fetchone()[0]} papers")
    return conn

def create_new_database():
    """Create a new empty database with schema"""
    if os.path.exists('papers.db'):
        print("Warning: Overwriting existing local database")
    
    print("Creating new database...")
    conn = create_database()
    print("Database created successfully")
    return conn

# Initialize database based on user choice
if db_choice == "create_new":
    print("Creating new database...")
    conn = create_new_database()
else:
    print("Loading existing database...")
    conn = load_existing_database()

# %% [markdown]
# ### 1.2.1 Database Schema
# This shows the schema used for both new and existing databases.

# %%
import sqlite3
import os

def create_database():
    """Create SQLite database with necessary tables"""
    conn = sqlite3.connect('papers.db')
    c = conn.cursor()
    
    # Print SQLite variable limit
    max_vars = get_sqlite_variable_limit(conn)
    print(f"SQLite max variables per query: {max_vars}")
    
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
    c.execute("CREATE INDEX IF NOT EXISTS idx_created ON papers(created)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_updated ON papers(updated)")
    
    conn.commit()
    return conn

# Initialize database
conn = create_database()

# %% [markdown]
# # 2. Data Collection

# %% [markdown]
# ## 2.1 Harvesting Configuration
# Configure the number of papers to fetch and provide a resumption token if continuing an interrupted harvest.
#
# ### Number of Papers
# - Set to 0 to fetch all CS papers (warning: this will take several hours)
# - Set to a specific number (e.g., 100) to test the harvesting process
#
# ### Resumption Token
# If harvesting was interrupted (e.g., due to timeout or error), you can continue from where you left off:
# 1. Copy the resumption token from the error message or last output
# 2. Paste it below to resume harvesting from that point
# 3. You can paste either:
#    - The full URL (e.g., `http://export.arxiv.org/oai2?verb=ListRecords&resumptionToken=foo%7Cbar`)
#    - Just the token part (e.g., `foo|bar`)
#    - URL-encoded characters (like %7C) will be automatically converted

# %%
# @title Harvesting Configuration  {"run":"auto"}
num_papers = 0 # @param {type:"slider", min:0, max:10000, step:100}
resumption_token = "" # @param {type:"string"}

# %%
from sickle import Sickle
from sickle.models import Record
from tqdm import tqdm
import urllib.parse
import re

def get_sqlite_variable_limit(conn):
    """Get the maximum number of variables allowed in a SQLite query"""
    c = conn.cursor()
    c.execute('PRAGMA compile_options')
    compile_options = c.fetchall()
    for option in compile_options:
        if 'MAX_VARIABLE_NUMBER=' in option[0]:
            return int(option[0].split('=')[1])
    return 999  # Default SQLite limit if not found

def get_safe_batch_size(conn, vars_per_item=1):
    """Calculate a safe batch size based on SQLite's variable limit
    
    Args:
        conn: SQLite connection
        vars_per_item: Number of variables needed per item in a batch
        
    Returns:
        int: Safe batch size that won't exceed SQLite's variable limit
    """
    max_vars = get_sqlite_variable_limit(conn)
    # Use 90% of the limit to be safe
    safe_vars = int(max_vars * 0.9)
    return safe_vars // vars_per_item

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
    
    print(f"\nSaving {len(papers)} papers and their authors...")
    with tqdm(total=len(papers), desc="Saving papers", unit=" papers",
              miniters=500,
              smoothing=0.8
              ) as pbar:
        for paper in papers:
            try:
                # Insert paper with all fields
                c.execute('''
                    INSERT OR IGNORE INTO papers (
                        id, title, abstract, categories,
                        msc_class, acm_class, doi, license, comments
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
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
                
                # Process authors
                for pos, author in enumerate(paper['authors'], 1):
                    # Clean and normalize author fields
                    keyname = author['keyname'].strip()
                    forenames = author.get('forenames', '').strip() if author.get('forenames') else ''
                    suffix = normalize_suffix(author.get('suffix'))
                    
                    # First try to find existing author with normalized fields
                    c.execute('''
                        SELECT id FROM authors 
                        WHERE keyname = ? AND 
                              COALESCE(forenames, '') = ? AND
                              COALESCE(suffix, '') = COALESCE(?, '')
                    ''', (
                        keyname,
                        forenames,
                        suffix
                    ))
                    result = c.fetchone()
                    
                    if result:
                        author_id = result[0]
                    else:
                        # Insert new author with normalized fields
                        c.execute('''
                            INSERT INTO authors (keyname, forenames, suffix)
                            VALUES (?, ?, ?)
                        ''', (
                            keyname,
                            forenames if forenames else None,  # Store NULL if empty
                            suffix  # Already normalized
                        ))
                        author_id = c.lastrowid
                    
                    # Create paper-author relationship
                    c.execute('''
                        INSERT OR IGNORE INTO paper_authors (paper_id, author_id, author_position)
                        VALUES (?, ?, ?)
                    ''', (paper['id'], author_id, pos))
                
                pbar.update(1)
                    
            except sqlite3.IntegrityError as e:
                print(f"Error saving paper {paper['id']}: {str(e)}")
                continue
                
    conn.commit()

def save_versions(paper_versions, conn):
    """Save paper version information to database efficiently"""
    c = conn.cursor()
    
    # Calculate batch sizes
    VERSION_BATCH_SIZE = get_safe_batch_size(conn, vars_per_item=5)  # 5 vars per version
    PAPER_UPDATE_BATCH_SIZE = get_safe_batch_size(conn, vars_per_item=4)  # 4 vars per paper update
    
    print(f"Using batch sizes:")
    print(f"- Versions: {VERSION_BATCH_SIZE}")
    print(f"- Paper Updates: {PAPER_UPDATE_BATCH_SIZE}")
    
    # Prepare batch data
    paper_updates = []
    version_inserts = []
    
    # Group versions by paper ID
    paper_version_map = {}
    for paper in paper_versions:
        paper_id = paper['id']
        if 'versions' in paper:
            paper_version_map[paper_id] = paper['versions']
    
    # Collect all data first
    for paper_id, versions in paper_version_map.items():
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
        for i in range(0, len(paper_updates), PAPER_UPDATE_BATCH_SIZE):
            batch = paper_updates[i:i + PAPER_UPDATE_BATCH_SIZE]
            c.executemany('''
                UPDATE papers 
                SET created = ?, updated = ?, withdrawn = ?
                WHERE id = ?
            ''', batch)
        
        # Batch insert versions
        for i in range(0, len(version_inserts), VERSION_BATCH_SIZE):
            batch = version_inserts[i:i + VERSION_BATCH_SIZE]
            c.executemany('''
                INSERT OR IGNORE INTO paper_versions (
                    paper_id, version, source_type, size, date
                ) VALUES (?, ?, ?, ?, ?)
            ''', batch)
        
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
        with tqdm(desc=f"Fetching {metadata_prefix}", unit=" papers",
                 miniters=25,
                 smoothing=0.8
                 ) as pbar:
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
# ## 2.2 Core Metadata Collection
# Fetches primary metadata (titles, abstracts, authors) using arXiv's OAI-PMH API.
# You can run this section independently and save before proceeding to version metadata.

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
# ## 2.3 Version Metadata Collection
# Fetches additional metadata about paper versions using arXiv's OAI-PMH API with arXivRaw format.
# **Note**: Run this after collecting core metadata, and ensure database is loaded if in a new session.

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
# # 3. Data Quality and Management

# %% [markdown]
# ## 3.1 Quality Check
# Performs comprehensive analysis of the collected data, including:
# - Sample paper inspection
# - Coverage statistics
# - Author statistics
# - Category distribution
# - Duplicate detection

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
    
    # Withdrawn paper statistics
    c.execute('SELECT COUNT(*) FROM papers WHERE withdrawn = 1')
    withdrawn_count = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM papers WHERE withdrawn = 0')
    active_count = c.fetchone()[0]
    print(f"Active Papers: {active_count} ({(active_count/total_papers*100):.1f}% of total)")
    print(f"Withdrawn Papers: {withdrawn_count} ({(withdrawn_count/total_papers*100):.1f}% of total)")
    
    # CS Primary papers
    c.execute('''
        WITH split_categories AS (
            SELECT id, withdrawn,
                   TRIM(SUBSTR(categories, 1, INSTR(categories || ' ', ' ') - 1)) as primary_category
            FROM papers
            WHERE categories IS NOT NULL
        )
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN withdrawn = 0 THEN 1 ELSE 0 END) as active
        FROM split_categories 
        WHERE primary_category LIKE 'cs.%'
    ''')
    cs_stats = c.fetchone()
    cs_total, cs_active = cs_stats
    print(f"\nCS primary papers: {cs_total} ({(cs_total/total_papers*100):.1f}% of total)")
    print(f"Active CS primary papers: {cs_active} ({(cs_active/cs_total*100):.1f}% of CS papers)")
    
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

    # Duplicate title analysis
    def normalize_title(title):
        """Normalize title for comparison"""
        if not title:
            return ""
        # Convert to lowercase
        title = title.lower()
        # Remove version numbers
        title = re.sub(r'\bv\d+\b', '', title)
        # Remove arXiv identifiers
        title = re.sub(r'arxiv:\d+\.\d+', '', title, flags=re.IGNORECASE)
        # Remove punctuation and normalize whitespace
        title = re.sub(r'[^\w\s]', '', title)
        return ' '.join(title.split())
    
    print("\nAnalyzing duplicate titles...")
    c.execute('''
        SELECT 
            p.id,
            p.title,
            p.categories,
            GROUP_CONCAT(a.keyname || COALESCE(', ' || a.forenames, '') || COALESCE(' ' || a.suffix, '')) as authors
        FROM papers p
        LEFT JOIN paper_authors pa ON p.id = pa.paper_id
        LEFT JOIN authors a ON pa.author_id = a.id
        WHERE p.title IS NOT NULL AND p.withdrawn = 0  -- Only analyze active papers
        GROUP BY p.id
    ''')
    papers = c.fetchall()
    
    # Group by normalized title
    title_groups = {}
    for paper in papers:
        norm_title = normalize_title(paper[1])
        if norm_title in title_groups:
            title_groups[norm_title].append(paper)
        else:
            title_groups[norm_title] = [paper]
    
    # Find groups with multiple papers
    duplicates = {title: papers for title, papers in title_groups.items() if len(papers) > 1}
    
    print(f"\nDuplicate Title Analysis (Active Papers Only):")
    print(f"Found {len(duplicates)} groups of papers with identical normalized titles")
    total_dupes = sum(len(papers) for papers in duplicates.values())
    print(f"Total papers involved in duplicates: {total_dupes}")
    
    # Print some examples
    if duplicates:
        print("\nExample duplicate groups:")
        for title, group in list(duplicates.items())[:3]:  # Show first 3 groups
            print(f"\nNormalized Title: {title}")
            for paper in group:
                print(f"- ID: {paper[0]}")
                print(f"  Original Title: {paper[1]}")
                print(f"  Categories: {paper[2]}")
                print(f"  Authors: {paper[3]}")
            print("-" * 80)

# Inspect sample papers
inspect_papers(conn) 

# %% [markdown]
# ## 3.2 Database Backup
# Saves the database to Google Drive for persistence.

# %%
# Mount Google Drive
from google.colab import drive # pyright: ignore [reportMissingImports]
drive.mount('/content/drive')

# Create project directory if it doesn't exist
%mkdir -p "/content/drive/MyDrive/ai-safety-papers" # pyright: ignore

# Copy database to Drive
%cp papers.db "/content/drive/MyDrive/ai-safety-papers/papers.db" # pyright: ignore
print("Database saved to Google Drive at: /ai-safety-papers/papers.db")

# %% [markdown]
# ## 3.3 Database Recovery
# Use this section if harvesting was interrupted and you need to clean up and retry with data in memory.
# **Important**: Only run if you have the `initial_papers` variable from a previous interrupted run.

# %%
# Drop all tables in correct order (respecting foreign key constraints)
c = conn.cursor()
c.execute("DROP TABLE IF EXISTS paper_versions")
c.execute("DROP TABLE IF EXISTS paper_authors")
c.execute("DROP TABLE IF EXISTS authors")
c.execute("DROP TABLE IF EXISTS papers")
conn.commit()

# Recreate tables
create_database()

# Resave papers from memory
if 'initial_papers' in locals():
    print("Found papers in memory, saving them...")
    save_papers(initial_papers, conn)
    
    # Print count of saved papers
    c.execute('SELECT COUNT(*) FROM papers')
    final_count = c.fetchone()[0]
    print(f"\nPapers saved after cleanup: {final_count}")
else:
    print("No papers found in memory. You'll need to run the harvesting cell again.") 