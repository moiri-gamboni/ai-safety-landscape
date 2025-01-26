# %% [markdown]
# # AI Safety Papers - Abstract Embedding Phase
# 
# This notebook generates Voyage AI embeddings for paper abstracts and stores them in the database.

# %% [markdown]
# ## 1. Environment Setup

# %%
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository if running in Colab
import os
if 'COLAB_GPU' in os.environ:
    # Clone the repository
    !git clone https://github.com/moiri-gamboni/ai-safety-landscape.git
    %cd ai-safety-landscape

# Install required packages
%pip install -r requirements.txt

# %% [markdown]
# ## 2. Database Connection

# %%
import sqlite3
import os
import numpy as np
from tqdm import tqdm
import voyageai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Tuple, Optional

# Load API key - try Colab form first, then .env
if 'COLAB_GPU' in os.environ:
    print("Getting Voyage AI API key from Colab form")
    # @title Voyage AI API Key
    voyage_api_key = "" # @param {type:"string"}
    # Set it for the client
    voyageai.api_key = voyage_api_key
else:
    print("Running locally, loading API key from .env")
    load_dotenv()

# Initialize Voyage AI client
vo = voyageai.Client()

# Path to database
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"
local_db = "papers.db"

# Copy database to local storage if needed
print(f"Copying database to local storage: {local_db}")
if not os.path.exists(local_db):
    %cp "{db_path}" {local_db}

conn = sqlite3.connect(local_db)
conn.row_factory = sqlite3.Row

# Check if abstract_embedding and token_count columns exist
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(papers)")
columns = [column[1] for column in cursor.fetchall()]

if 'abstract_embedding' not in columns:
    print("Adding abstract_embedding column...")
    conn.execute('''
        ALTER TABLE papers 
        ADD COLUMN abstract_embedding BLOB
    ''')
    conn.commit()

if 'token_count' not in columns:
    print("Adding token_count column...")
    conn.execute('''
        ALTER TABLE papers 
        ADD COLUMN token_count INTEGER
    ''')
    conn.commit()

# %% [markdown]
# ## 3. Embedding Generation

# %%
# Configuration
max_batches = 1  # Set to None to process all batches, or a number to limit processing
batch_size = 128  # Maximum batch size for Voyage AI

# Rate limit tracking
class RateLimiter:
    def __init__(self, rpm_limit: int = 2000, tpm_limit: int = 8_000_000, window_seconds: int = 60):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        self.requests = []
        self.tokens = []
    
    def add_request(self, token_count: int = 0):
        current_time = time.time()
        # Clean up old entries
        cutoff_time = current_time - self.window_seconds
        self.requests = [t for t in self.requests if t > cutoff_time]
        self.tokens = [t for t, _ in self.tokens if t > cutoff_time]
        
        # Add new request
        self.requests.append(current_time)
        if token_count > 0:
            self.tokens.append((current_time, token_count))
    
    def can_make_request(self, token_count: int = 0) -> Tuple[bool, Optional[float]]:
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Clean up old entries
        self.requests = [t for t in self.requests if t > cutoff_time]
        self.tokens = [(t, c) for t, c in self.tokens if t > cutoff_time]
        
        # Check RPM limit
        if len(self.requests) >= self.rpm_limit:
            wait_time = self.requests[0] - cutoff_time
            return False, wait_time
        
        # Check TPM limit
        current_tokens = sum(count for _, count in self.tokens)
        if current_tokens + token_count > self.tpm_limit:
            # Find the earliest time when we'll be under the limit
            sorted_tokens = sorted(self.tokens)
            running_sum = current_tokens + token_count
            for t, c in sorted_tokens:
                running_sum -= c
                if running_sum <= self.tpm_limit:
                    wait_time = t - cutoff_time
                    return False, wait_time
        
        return True, None

rate_limiter = RateLimiter()

def get_csai_papers(conn: sqlite3.Connection, batch_size: int = 128):
    """Generator that yields batches of papers with cs.AI in their categories
    
    Args:
        conn: Database connection
        batch_size: Maximum number of papers per batch
    """
    cursor = conn.cursor()
    
    # First get total count for progress bar
    cursor.execute('''
        WITH split_categories AS (
            SELECT id, abstract
            FROM papers
            WHERE categories LIKE '%cs.AI%'
              AND abstract IS NOT NULL
              AND abstract_embedding IS NULL
        )
        SELECT COUNT(*) FROM split_categories
    ''')
    total_papers = cursor.fetchone()[0]
    
    # Then fetch papers in batches
    cursor.execute('''
        WITH split_categories AS (
            SELECT id, abstract
            FROM papers
            WHERE categories LIKE '%cs.AI%'
              AND abstract IS NOT NULL
              AND abstract_embedding IS NULL
        )
        SELECT id, abstract FROM split_categories
    ''')
    
    batch = []
    with tqdm(total=total_papers, desc="Processing papers", unit=" papers") as pbar:
        for row in cursor:
            batch.append((row['id'], row['abstract']))
            if len(batch) >= batch_size:
                pbar.update(len(batch))
                yield batch
                batch = []
        if batch:
            pbar.update(len(batch))
            yield batch

def adjust_batch_for_token_limit(batch: List[Tuple[str, str]], model: str = "voyage-3-large") -> List[List[Tuple[str, str]]]:
    """Split a batch into sub-batches that respect the token limit
    
    Args:
        batch: List of (id, abstract) tuples
        model: Voyage AI model to use
    
    Returns:
        List of batches, each respecting the token limit
    """
    TOKEN_LIMIT = 120_000  # voyage-3-large limit
    
    abstracts = [item[1] for item in batch]
    token_counts = [vo.count_tokens([abstract], model=model) for abstract in abstracts]
    
    sub_batches = []
    current_batch = []
    current_tokens = 0
    
    for (paper_id, abstract), token_count in zip(batch, token_counts):
        # If single abstract exceeds limit, skip it
        if token_count > TOKEN_LIMIT:
            print(f"Warning: Abstract {paper_id} exceeds token limit ({token_count} tokens), skipping")
            continue
            
        # If adding this abstract would exceed limit, start new batch
        if current_tokens + token_count > TOKEN_LIMIT:
            if current_batch:
                sub_batches.append(current_batch)
            current_batch = [(paper_id, abstract)]
            current_tokens = token_count
        else:
            current_batch.append((paper_id, abstract))
            current_tokens += token_count
    
    if current_batch:
        sub_batches.append(current_batch)
    
    return sub_batches

def process_batch(batch: List[Tuple[str, str]], model: str = "voyage-3-large") -> List[Tuple[str, List[float], int]]:
    """Process a batch of papers, returning embeddings and token counts
    
    Args:
        batch: List of (id, abstract) tuples
        model: Voyage AI model to use
    
    Returns:
        List of (id, embedding, token_count) tuples
    """
    abstracts = [item[1] for item in batch]
    
    # First count tokens
    token_count = vo.count_tokens(abstracts, model=model)
    
    # Check rate limits and wait if necessary
    can_proceed, wait_time = rate_limiter.can_make_request(token_count)
    if not can_proceed:
        print(f"\nRate limit reached, waiting {wait_time:.1f} seconds...")
        time.sleep(wait_time + 0.1)  # Add small buffer
    
    # Generate embeddings with 2048 dimensions
    result = vo.embed(
        abstracts,
        model=model,
        input_type="document",
        output_dimension=2048
    )
    
    # Update rate limiter
    rate_limiter.add_request(token_count)
    
    # Return list of (id, embedding, token_count) tuples
    return list(zip(
        [item[0] for item in batch],
        result.embeddings,
        [token_count] * len(batch)  # Each abstract's token count
    ))

# Process in batches
total_updated = 0
batches_processed = 0

try:
    for batch in get_csai_papers(conn, batch_size):
        if max_batches is not None and batches_processed >= max_batches:
            print(f"\nStopping after {max_batches} batch(es) as requested")
            break
            
        # Split batch if needed to respect token limit
        sub_batches = adjust_batch_for_token_limit(batch)
        
        for sub_batch in sub_batches:
            try:
                # Process batch and get embeddings
                results = process_batch(sub_batch)
                
                # Store in database
                cursor = conn.cursor()
                for paper_id, embedding, token_count in results:
                    embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                    cursor.execute('''
                        UPDATE papers
                        SET abstract_embedding = ?,
                            token_count = ?
                        WHERE id = ?
                    ''', (embedding_blob, token_count, paper_id))
                
                conn.commit()
                total_updated += len(sub_batch)
                
            except Exception as e:
                print(f"\nError processing batch: {str(e)}")
                continue
        
        batches_processed += 1
        if max_batches is not None:
            print(f"\nProcessed {batches_processed}/{max_batches} batch(es)")

except Exception as e:
    print(f"\nFatal error in embedding generation: {str(e)}")
    raise
finally:
    print(f"\nProcess completed:")
    print(f"- Total papers processed: {total_updated}")
    print(f"- Total batches processed: {batches_processed}")

# %% [markdown]
# ## 4. Data Quality Validation

# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def validate_embeddings(conn):
    """Run quality checks on generated embeddings"""
    # Set fixed seeds for reproducibility
    np.random.seed(42)
    
    cursor = conn.cursor()
    
    print("\n=== Embedding Quality Checks ===")
    
    # 1. Coverage Check
    cursor.execute('''
        SELECT 
            COUNT(*) AS total_csai,
            SUM(CASE WHEN abstract_embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding,
            SUM(CASE WHEN abstract_embedding IS NULL THEN 1 ELSE 0 END) AS without_embedding
        FROM papers 
        WHERE categories LIKE '%cs.AI%'
          AND abstract IS NOT NULL
    ''')
    stats = cursor.fetchone()
    print(f"\n1. Coverage:")
    print(f"- Total CS.AI papers: {stats['total_csai']}")
    print(f"- With embeddings: {stats['with_embedding']} ({(stats['with_embedding']/stats['total_csai'])*100:.1f}%)")
    print(f"- Missing embeddings: {stats['without_embedding']} ({(stats['without_embedding']/stats['total_csai'])*100:.1f}%)")

    # 2. Validity Check (NaN/Zero vectors)
    cursor.execute('''
        SELECT abstract_embedding 
        FROM papers 
        WHERE abstract_embedding IS NOT NULL
    ''')
    invalid_count = 0
    total_checked = 0
    for row in cursor:
        embedding = np.frombuffer(row['abstract_embedding'], dtype=np.float32)
        if np.isnan(embedding).any():
            invalid_count += 1
        elif np.all(embedding == 0):
            invalid_count += 1
        total_checked += 1
    
    print(f"\n2. Validity (sampled {total_checked}):")
    print(f"- Invalid embeddings (NaN/zeros): {invalid_count} ({(invalid_count/total_checked)*100:.1f}%)")

    # 3. Norm Analysis
    cursor.execute('''
        SELECT abstract_embedding 
        FROM papers 
        WHERE abstract_embedding IS NOT NULL
    ''')
    norms = []
    for row in cursor:
        embedding = np.frombuffer(row['abstract_embedding'], dtype=np.float32)
        norms.append(np.linalg.norm(embedding))
    
    print(f"\n3. Norm Analysis (sample):")
    print(f"- Mean norm: {np.mean(norms):.2f}")
    print(f"- Std dev: {np.std(norms):.2f}")
    print(f"- Min/Max: {np.min(norms):.2f}/{np.max(norms):.2f}")

    # 4. Similarity Analysis
    # Get random pairs more efficiently
    cursor.execute('''
        SELECT id, title, abstract, abstract_embedding 
        FROM papers 
        WHERE abstract_embedding IS NOT NULL
          AND categories LIKE '%cs.AI%'
          AND ABS(RANDOM() % 100) = 0  -- Fast random sampling
        LIMIT 10000
    ''')
    papers = cursor.fetchall()
    random_embeddings = np.vstack([np.frombuffer(row['abstract_embedding'], dtype=np.float32) for row in papers])
    
    # Analyze embedding components
    print("\n4. Embedding Analysis:")
    print(f"- Shape: {random_embeddings.shape}")
    print("- Component statistics:")
    means = np.mean(random_embeddings, axis=0)
    stds = np.std(random_embeddings, axis=0)
    print(f"  Mean of means: {np.mean(means):.3f} ± {np.std(means):.3f}")
    print(f"  Mean of stds: {np.mean(stds):.3f} ± {np.std(stds):.3f}")
    print(f"  Component range: [{np.min(random_embeddings):.3f}, {np.max(random_embeddings):.3f}]")
    
    # Show distribution of components
    positive_frac = np.mean(random_embeddings > 0)
    print(f"  Fraction of positive components: {positive_frac:.3f}")
    
    # Compute similarities
    n_samples = len(random_embeddings)
    if n_samples > 1:
        # Generate random pairs of indices
        idx1 = np.random.randint(0, n_samples, size=1000)
        idx2 = np.random.randint(0, n_samples, size=1000)
        # Exclude self-pairs
        valid_pairs = idx1 != idx2
        idx1, idx2 = idx1[valid_pairs], idx2[valid_pairs]
        
        # Compute cosine similarities properly
        similarities = cosine_similarity(random_embeddings[idx1], random_embeddings[idx2]).diagonal()
        
        print("\n5. Similarity Analysis:")
        print(f"- Random pairs (n={len(similarities)}):")
        print(f"  Mean: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}")
        print(f"  Range: [{np.min(similarities):.3f}, {np.max(similarities):.3f}]")
        
        # Print example pairs with different similarity levels
        print("\nExample pairs:")
        # Most similar pair
        most_similar_idx = np.argmax(similarities)
        print(f"\nMost similar pair (similarity: {similarities[most_similar_idx]:.3f}):")
        print(f"Paper 1: {papers[idx1[most_similar_idx]]['title']}")
        print(f"URL 1: https://arxiv.org/abs/{papers[idx1[most_similar_idx]]['id']}")
        print(f"Abstract 1:\n{papers[idx1[most_similar_idx]]['abstract']}")
        print(f"\nPaper 2: {papers[idx2[most_similar_idx]]['title']}")
        print(f"URL 2: https://arxiv.org/abs/{papers[idx2[most_similar_idx]]['id']}")
        print(f"Abstract 2:\n{papers[idx2[most_similar_idx]]['abstract']}")
        
        # Least similar pair
        least_similar_idx = np.argmin(similarities)
        print(f"\nLeast similar pair (similarity: {similarities[least_similar_idx]:.3f}):")
        print(f"Paper 1: {papers[idx1[least_similar_idx]]['title']}")
        print(f"URL 1: https://arxiv.org/abs/{papers[idx1[least_similar_idx]]['id']}")
        print(f"Abstract 1:\n{papers[idx1[least_similar_idx]]['abstract']}")
        print(f"\nPaper 2: {papers[idx2[least_similar_idx]]['title']}")
        print(f"URL 2: https://arxiv.org/abs/{papers[idx2[least_similar_idx]]['id']}")
        print(f"Abstract 2:\n{papers[idx2[least_similar_idx]]['abstract']}")
        
        # Median similarity pair
        median_idx = np.argsort(similarities)[len(similarities)//2]
        print(f"\nMedian similarity pair (similarity: {similarities[median_idx]:.3f}):")
        print(f"Paper 1: {papers[idx1[median_idx]]['title']}")
        print(f"URL 1: https://arxiv.org/abs/{papers[idx1[median_idx]]['id']}")
        print(f"Abstract 1:\n{papers[idx1[median_idx]]['abstract']}")
        print(f"\nPaper 2: {papers[idx2[median_idx]]['title']}")
        print(f"URL 2: https://arxiv.org/abs/{papers[idx2[median_idx]]['id']}")
        print(f"Abstract 2:\n{papers[idx2[median_idx]]['abstract']}")
    else:
        print("\n5. Similarity Analysis:")
        print("Not enough samples for similarity analysis")
    
    # # Get potential duplicates (papers with same title)
    # cursor.execute('''
    #     SELECT p1.id as id1, p2.id as id2,
    #            p1.title as title1,
    #            p1.abstract as abstract1,
    #            p2.abstract as abstract2,
    #            p1.abstract_embedding as e1,
    #            p2.abstract_embedding as e2
    #     FROM papers p1
    #     JOIN papers p2 ON LOWER(TRIM(p1.title)) = LOWER(TRIM(p2.title))
    #     WHERE p1.id < p2.id  -- Avoid self-joins and duplicates
    #       AND p1.withdrawn = 0 AND p2.withdrawn = 0
    #       AND p1.abstract_embedding IS NOT NULL
    #       AND p2.abstract_embedding IS NOT NULL
    #       AND p1.categories LIKE '%cs.AI%'
    #       AND p2.categories LIKE '%cs.AI%'
    #     LIMIT 50
    # ''')
    
    # rows = cursor.fetchall()
    # if rows:
    #     # Process all pairs at once
    #     e1 = np.vstack([np.frombuffer(row['e1'], dtype=np.float32) for row in rows])
    #     e2 = np.vstack([np.frombuffer(row['e2'], dtype=np.float32) for row in rows])
    #     similarities = np.sum(e1 * e2, axis=1)
        
    #     print(f"\n6. Same-title Analysis:")
    #     print(f"- Same-title pairs (n={len(similarities)}):")
    #     print(f"  Mean: {np.mean(similarities):.3f} ± {np.std(similarities):.3f}")
    #     print(f"  Range: [{np.min(similarities):.3f}, {np.max(similarities):.3f}]")
        
    #     # Print example same-title pairs
    #     print("\nExample same-title pairs:")
    #     # Most similar pair
    #     most_similar_idx = np.argmax(similarities)
    #     print(f"\nMost similar pair (similarity: {similarities[most_similar_idx]:.3f}):")
    #     print(f"Title: {rows[most_similar_idx]['title1']}")
    #     print(f"URL 1: https://arxiv.org/abs/{rows[most_similar_idx]['id1']}")
    #     print(f"Abstract 1:\n{rows[most_similar_idx]['abstract1']}")
    #     print(f"\nURL 2: https://arxiv.org/abs/{rows[most_similar_idx]['id2']}")
    #     print(f"Abstract 2:\n{rows[most_similar_idx]['abstract2']}")
        
    #     # Least similar pair
    #     least_similar_idx = np.argmin(similarities)
    #     print(f"\nLeast similar pair (similarity: {similarities[least_similar_idx]:.3f}):")
    #     print(f"Title: {rows[least_similar_idx]['title1']}")
    #     print(f"URL 1: https://arxiv.org/abs/{rows[least_similar_idx]['id1']}")
    #     print(f"Abstract 1:\n{rows[least_similar_idx]['abstract1']}")
    #     print(f"\nURL 2: https://arxiv.org/abs/{rows[least_similar_idx]['id2']}")
    #     print(f"Abstract 2:\n{rows[least_similar_idx]['abstract2']}")
    # else:
    #     print("\n6. Same-title Analysis:")
    #     print("No same-title pairs found for comparison")

# Run validation
validate_embeddings(conn)

# %% [markdown]
# ## 5. Database Backup

# %%
# Copy updated database back to Drive
!cp {local_db} "{db_path}"
print("Database backup completed to Google Drive")

