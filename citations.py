# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # AI Safety Papers - Citation Tracking
# 
# This notebook fetches citation counts from OpenCitations for CS.AI papers in the database.

# %% [markdown]
# ## 1. Setup

# %%
# Mount Google Drive
from google.colab import drive # pyright: ignore [reportMissingImports]
drive.mount('/content/drive')

# Install required packages if running in Colab
import os
if 'COLAB_GPU' in os.environ:
    !sudo apt-get -qq update && sudo apt-get -qq install postgresql postgresql-contrib # pyright: ignore
    !sudo service postgresql start # pyright: ignore
    !sudo sed -i 's/local\s*all\s*postgres\s*peer/local all postgres trust/' /etc/postgresql/14/main/pg_hba.conf # pyright: ignore
    !sudo service postgresql restart # pyright: ignore
    
    %pip install psycopg2-binary requests tenacity tqdm # pyright: ignore

# %% [markdown]
# ## 2. Load Database

# %%
import psycopg2

def get_db_connection():
    """Create PostgreSQL connection"""
    return psycopg2.connect(
        host='',
        database="postgres",
        user="postgres"
    )

def load_database():
    """Load PostgreSQL backup using psql"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/papers_postgres.sql"
    print("Loading PostgreSQL backup...")
    !psql -U postgres -d postgres -f "{backup_path}" # pyright: ignore

load_database()
conn = get_db_connection()

# %% [markdown]
# ## 3. Setup Citation Column

# %%
def setup_citation_column():
    with conn.cursor() as cursor:
        # Create column fresh with NULL default
        cursor.execute('''
            ALTER TABLE papers 
            ADD COLUMN IF NOT EXISTS citation_count INTEGER DEFAULT NULL
        ''')
        conn.commit()

setup_citation_column()

# %% [markdown]
# ## 4. Async Citation Fetching

# %%
import aiohttp
import asyncio
from tqdm import tqdm

# Load OpenCitations API key
if 'COLAB_GPU' in os.environ:
    # @title OpenCitations API Key
    oc_token = "" # @param {type:"string"}
else:
    from dotenv import load_dotenv
    load_dotenv()
    oc_token = os.getenv('OPENCITATIONS_ACCESS_TOKEN')

API_HEADERS = {"authorization": oc_token} if oc_token else {}
BASE_URL = "https://opencitations.net/index/api/v2/citation-count/doi:"

def arxiv_id_to_doi(arxiv_id: str) -> str:
    """Convert arXiv ID to DataCite DOI format"""
    return f"10.48550/arXiv.{arxiv_id}"

# Configure async parameters
CONCURRENCY_LIMIT = 8  # OpenCitations recommends 10 req/s
BATCH_SIZE = 1000  # Papers per progress update

# %% [markdown]
# ## 5. Async Citation Fetching

# %%
from tenacity import retry, stop_after_attempt, wait_exponential

async def fetch_all_citations():
    """Top-level async function for notebook execution"""
    async with aiohttp.ClientSession(
        headers=API_HEADERS,
        connector=aiohttp.TCPConnector(limit=CONCURRENCY_LIMIT),
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        # Get all paper IDs needing processing
        with conn.cursor() as cursor:
            cursor.execute('''
                SELECT id FROM papers 
                WHERE categories LIKE '%cs.AI%'
                  AND withdrawn = FALSE
                  AND citation_count IS NULL
            ''')
            paper_ids = [row[0] for row in cursor.fetchall()]

        # Process with progress tracking
        with tqdm(total=len(paper_ids), desc="Fetching citations") as pbar:
            semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
            
            async def process_with_semaphore(paper_id):
                async with semaphore:
                    await process_paper(session, paper_id)
                    pbar.update(1)
            
            # Batch processing for memory management
            for i in range(0, len(paper_ids), BATCH_SIZE):
                batch = paper_ids[i:i+BATCH_SIZE]
                await asyncio.gather(*[process_with_semaphore(pid) for pid in batch])
                del batch  # Explicit memory cleanup
                await asyncio.sleep(1)  # Rate limit between batches

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
async def fetch_citation_count(session, doi: str) -> int:
    """Async fetch with retries and proper error handling"""
    try:
        async with session.get(f"{BASE_URL}{doi}") as response:
            response.raise_for_status()
            data = await response.json()
            return int(data[0]['count'])
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Network error for {doi}: {str(e)}")
        raise
    except (IndexError, KeyError, ValueError) as e:
        print(f"Invalid response for {doi}: {str(e)}")
        return 0  # Treat as valid zero-citation response

async def process_paper(session, paper_id: str):
    """Process single paper including DB update"""
    doi = arxiv_id_to_doi(paper_id)
    try:
        count = await fetch_citation_count(session, doi)
        # Run DB update in thread pool
        await asyncio.to_thread(
            update_citation_in_db,
            paper_id,
            count
        )
    except Exception as e:
        print(f"Failed processing {paper_id}: {str(e)}")

def update_citation_in_db(paper_id: str, count: int):
    """Synchronous DB update function"""
    with conn.cursor() as cursor:
        cursor.execute('''
            UPDATE papers 
            SET citation_count = %s 
            WHERE id = %s
        ''', (count, paper_id))
        conn.commit()

await fetch_all_citations() # pyright: ignore

# %% [markdown]
# ## 5. Data Validation

# %%
def validate_citations():
    """Validate citation data quality"""
    with conn.cursor() as cursor:
        # Track processing status
        cursor.execute('''
            SELECT 
                COUNT(*) FILTER (WHERE citation_count IS NOT NULL) AS processed,
                COUNT(*) FILTER (WHERE citation_count IS NULL) AS unprocessed
            FROM papers 
            WHERE categories LIKE '%cs.AI%'
              AND withdrawn = FALSE
        ''')
        processed, unprocessed = cursor.fetchone()
        
        print(f"\nProcessing Status:")
        print(f"• Processed papers: {processed}")
        print(f"• Remaining unprocessed: {unprocessed}")

        # Only show stats for processed papers
        if processed > 0:
            cursor.execute('''
                SELECT 
                    AVG(citation_count) AS mean,
                    STDDEV(citation_count) AS stddev,
                    MIN(citation_count) AS min,
                    MAX(citation_count) AS max
                FROM papers 
                WHERE categories LIKE '%cs.AI%'
                  AND withdrawn = FALSE
                  AND citation_count IS NOT NULL
            ''')
            stats = cursor.fetchone()
            print(f"\nCitation Statistics for {processed} processed papers:")
            print(f"• Average: {stats[0]:.1f} ± {stats[1]:.1f}")
            print(f"• Range: {stats[2]} - {stats[3]}")

validate_citations()

# %% [markdown]
# ## 6. Save Results

# %%
def backup_citations():
    """Use pg_dump for PostgreSQL backups"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/papers_postgres.sql"
    !pg_dump -U postgres -F p -f "{backup_path}" postgres # pyright: ignore

# Call backup after processing
backup_citations()