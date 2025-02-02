# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # AI Safety Papers - Embedding Phase
# 
# This notebook generates Voyage AI embeddings for paper titles and abstracts and stores them in the database.

# %% [markdown]
# ## 1. Environment Setup

# %%
# Mount Google Drive
from google.colab import drive # pyright: ignore [reportMissingImports]
drive.mount('/content/drive')

# Install required packages if running in Colab
import os
if 'COLAB_GPU' in os.environ:
    # Install and configure PostgreSQL
    !sudo apt-get -qq update && sudo apt-get -qq install postgresql postgresql-contrib # pyright: ignore
    !sudo service postgresql start # pyright: ignore
    !sudo sed -i 's/local\s*all\s*postgres\s*peer/local all postgres trust/' /etc/postgresql/14/main/pg_hba.conf # pyright: ignore
    !sudo service postgresql restart # pyright: ignore
    
    # Install Python client
    %pip install psycopg2-binary # pyright: ignore
    %pip install voyageai tqdm scikit-learn tenacity # pyright: ignore

# %% [markdown]
# ## 2. Database Connection

# %%
import psycopg2
import os
import numpy as np
from tqdm import tqdm
import voyageai
import time
from typing import List, Tuple, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import re
from psycopg2.extras import DictCursor
from cuml.preprocessing import StandardScaler
from cuml.neighbors import NearestNeighbors

# Load API key from Colab form or environment
if 'COLAB_GPU' in os.environ:
    print("Getting Voyage AI API key from Colab form")
    # @title Voyage AI API Key
    voyage_api_key = "" # @param {type:"string"}
    # Set it for the client
    voyageai.api_key = voyage_api_key
else:
    from dotenv import load_dotenv
    print("Running locally, loading API key from .env")
    load_dotenv()
    voyageai.api_key = os.getenv('VOYAGE_API_KEY')

# Initialize Voyage AI client
vo = voyageai.Client()

# Path to database
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"

def load_database():
    """Load PostgreSQL backup using psql"""
    print("Loading PostgreSQL backup...")
    !createdb -U postgres papers # pyright: ignore
    !pg_restore -U postgres --jobs=8 -d papers "{backup_path}" # pyright: ignore

def connect_db():
    """Connect to PostgreSQL database with schema validation"""
    conn = psycopg2.connect(
        host='',
        database="papers",
        user="postgres"
    )
    return conn

load_database()
conn = connect_db()

# %% [markdown]
# ### Column Reset (Run when reprocessing)
# %%
def create_column():
    # Check for embedding column
    with conn.cursor() as cursor:
        print("Adding embedding column...")
        cursor.execute('''
            ALTER TABLE papers 
            ADD COLUMN IF NOT EXISTS embedding BYTEA
        ''')
        cursor.execute('''
            ALTER TABLE papers 
            ADD COLUMN IF NOT EXISTS scaled_embedding BYTEA
        ''')
        conn.commit()

create_column()

# %% [markdown]
# ## 3. Embedding Generation

# %%
# Configuration
max_batches = None  # Set to None to process all batches, or a number to limit processing
batch_size = 128  # Maximum batch size for Voyage AI

# Rate limit tracking
class RateLimiter:
    def __init__(self, rpm_limit=2000, tpm_limit=3_000_000):  # voyage-3-large limits: 2000 RPM, 3M TPM
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.requests = []  # List of timestamps
        self.tokens = []    # List of (timestamp, token_count) tuples
        
    def can_make_request(self, token_count):
        current_time = time.time()
        cutoff_time = current_time - 60  # 1 minute ago
        
        # Clean up old entries
        self.requests = [t for t in self.requests if t > cutoff_time]
        self.tokens = [(t, c) for t, c in self.tokens if t > cutoff_time]
        
        # Check RPM limit
        if len(self.requests) >= self.rpm_limit:
            print(f"\nRate limit reached at {time.strftime('%H:%M:%S')}:")
            print(f"- Request limit: {len(self.requests)}/{self.rpm_limit} RPM")
            return False
            
        # Check TPM limit
        total_tokens = sum(count for _, count in self.tokens)
        if total_tokens + token_count > self.tpm_limit:
            print(f"\nRate limit reached at {time.strftime('%H:%M:%S')}:")
            print(f"- Token limit: {total_tokens + token_count}/{self.tpm_limit} TPM")
            return False
            
        return True
        
    def add_request(self, token_count):
        current_time = time.time()
        self.requests.append(current_time)
        self.tokens.append((current_time, token_count))

rate_limiter = RateLimiter()

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def embed_with_backoff(texts, model="voyage-3-large"):
    return vo.embed(
        texts,
        model=model,
        input_type="document",
        output_dimension=2048
    )

def get_csai_papers(conn: psycopg2.extensions.connection, batch_size: int = 128):
    """Generator that yields batches of papers needing embeddings"""
    cursor = conn.cursor(cursor_factory=DictCursor)
    
    # First get total count for progress bar
    cursor.execute('''
        WITH relevant_papers AS (
            SELECT id, title, abstract, llm_category
            FROM papers
            WHERE title IS NOT NULL
              AND abstract IS NOT NULL
              AND embedding IS NULL
        )
        SELECT COUNT(*) FROM relevant_papers
    ''')
    total_papers = cursor.fetchone()[0]
    
    # Then fetch papers in batches
    cursor.execute('''
        WITH relevant_papers AS (
            SELECT id, title, abstract, llm_category
            FROM papers
            WHERE title IS NOT NULL
              AND abstract IS NOT NULL
              AND embedding IS NULL
        )
        SELECT id, title, abstract, llm_category FROM relevant_papers
    ''')
    
    batch = []
    with tqdm(total=total_papers, desc="Processing papers", unit=" papers") as pbar:
        for row in cursor:
            # Combine category with title/abstract
            combined_text = f"Category: {row['llm_category']}\n{row['title']}\n{row['abstract']}"
            batch.append((row['id'], combined_text))
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
        batch: List of (id, text) tuples
        model: Voyage AI model to use
    
    Returns:
        List of batches, each respecting the token limit
    """
    TOKEN_LIMIT = 120_000  # voyage-3-large limit
    
    texts = [item[1] for item in batch]
    token_counts = [vo.count_tokens([text], model=model) for text in texts]
    
    sub_batches = []
    current_batch = []
    current_tokens = 0
    
    for (paper_id, text), token_count in zip(batch, token_counts):
        # If single text exceeds limit, skip it
        if token_count > TOKEN_LIMIT:
            print(f"Warning: Text {paper_id} exceeds token limit ({token_count} tokens), skipping")
            continue
            
        # If adding this text would exceed limit, start new batch
        if current_tokens + token_count > TOKEN_LIMIT:
            if current_batch:
                sub_batches.append(current_batch)
            current_batch = [(paper_id, text)]
            current_tokens = token_count
        else:
            current_batch.append((paper_id, text))
            current_tokens += token_count
    
    if current_batch:
        sub_batches.append(current_batch)
    
    return sub_batches

def process_batch(batch: List[Tuple[str, str]], model: str = "voyage-3-large") -> List[Tuple[str, List[float]]]:
    """Process a batch of papers, returning embeddings"""
    texts = [item[1] for item in batch]
    paper_ids = [item[0] for item in batch]
    
    # First count tokens for rate limiting
    try:
        token_count = vo.count_tokens(texts, model=model)
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        raise
    
    # Check rate limits
    if not rate_limiter.can_make_request(token_count):
        raise Exception("Rate limit exceeded")  # This will trigger exponential backoff
    
    # Generate embeddings with exponential backoff
    try:
        result = embed_with_backoff(texts, model=model)
        
        if isinstance(result, float):
            raise ValueError(f"API returned float instead of EmbeddingsObject: {result}")
            
        if not hasattr(result, 'embeddings'):
            raise ValueError(f"Unexpected response from vo.embed: {result}")
            
        # Return list of (paper_id, embedding) tuples and total tokens used
        embeddings = result.embeddings
        rate_limiter.add_request(token_count)
        
        return [(paper_id, embedding) for paper_id, embedding in zip(paper_ids, embeddings)], result.total_tokens
            
    except Exception as e:
        print("\nError in vo.embed call:")
        print(f"Exception type: {type(e)}")
        print(f"Exception args: {e.args}")
        print(f"Full exception: {repr(e)}")
        print("\nInput that caused error:")
        print(f"First text: {texts[0][:500]}...")  # Truncate long texts
        raise

def scale_and_store_embeddings():
    """Scale embeddings using StandardScaler and store in database"""
    print("\nScaling embeddings...")
    
    # Load all raw embeddings
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute('SELECT id, embedding FROM papers WHERE embedding IS NOT NULL')
    papers = cursor.fetchall()
    
    if not papers:
        print("No embeddings found to scale")
        return
    
    # Convert to numpy array
    embeddings = np.array([np.frombuffer(p['embedding'], dtype=np.float32) for p in papers])
    
    # Scale embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)
    
    # Store scaled embeddings
    with conn.cursor() as update_cursor:
        for p, scaled_emb in zip(papers, scaled_embeddings):
            update_cursor.execute('''
                UPDATE papers
                SET scaled_embedding = %s
                WHERE id = %s
            ''', (scaled_emb.astype(np.float32).tobytes(), p['id']))
        conn.commit()
    print(f"Scaled {len(papers)} embeddings")

def generate_embeddings():
    # Process in batches
    total_updated = 0
    batches_processed = 0
    total_tokens = 0

    try:
        for batch in get_csai_papers(conn, batch_size):
            if max_batches is not None and batches_processed >= max_batches:
                print(f"\nStopping after {max_batches} batch(es) as requested")
                break
                
            # Split batch if needed to respect token limit
            sub_batches = adjust_batch_for_token_limit(batch)
            
            for i, sub_batch in enumerate(sub_batches):
                try:
                    # Process batch and get embeddings
                    results, batch_tokens = process_batch(sub_batch)
                    if not isinstance(results, list):
                        print(f"Warning: results is not a list, got {type(results)}")
                        print(f"Results value: {results}")
                        raise ValueError(f"Expected list result, got {type(results)}")
                    
                    # Store in database
                    cursor = conn.cursor()
                    for paper_id, embedding in results:
                        embedding_blob = np.array(embedding, dtype=np.float32).tobytes()
                        cursor.execute('''
                            UPDATE papers
                            SET embedding = %s
                            WHERE id = %s
                        ''', (embedding_blob, paper_id))
                    
                    conn.commit()
                    total_updated += len(sub_batch)
                    
                    # Update total tokens
                    total_tokens += batch_tokens
                    
                except Exception as e:
                    print(f"\nFatal error processing sub-batch:")
                    print(f"Exception type: {type(e)}")
                    print(f"Exception args: {e.args}")
                    print(f"Full exception: {repr(e)}")
                    print(f"Sub-batch contents: {sub_batch[:2]}...")
                    raise  # Stop execution here
            
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
        print(f"- Total tokens processed: {total_tokens:,}")

generate_embeddings()
scale_and_store_embeddings()

# %% [markdown]
# ## 5. KNN Graph Generation

# %%
def generate_knn_graph() -> tuple[np.ndarray, np.ndarray]:
    """Generate KNN graph for clustering"""
    print("\nGenerating KNN graph...")
    
    # Load all scaled embeddings
    cursor = conn.cursor(cursor_factory=DictCursor)
    cursor.execute('SELECT id, scaled_embedding FROM papers WHERE scaled_embedding IS NOT NULL')
    papers = cursor.fetchall()
    
    if not papers:
        raise ValueError("No scaled embeddings found in database")
    
    # Convert to numpy array
    embeddings = np.array([np.frombuffer(row['scaled_embedding'], dtype=np.float32) for row in papers])
    
    # Create KNN graph
    print("Computing 100-NN graph...")
    nn_model = NearestNeighbors(n_neighbors=100, metric='cosine')
    nn_model.fit(embeddings)
    distances, indices = nn_model.kneighbors(embeddings)
    
    # Save to Drive
    save_path = "/content/drive/MyDrive/ai-safety-papers/knn_graph.npz"
    np.savez_compressed(save_path, distances=distances, indices=indices)
    print(f"Saved KNN graph to {save_path}")
    
    return distances, indices

distances, indices = generate_knn_graph()

# %% [markdown]
# ## 6. Validation Metrics

# %%
def validate_embeddings(distances: np.ndarray, indices: np.ndarray) -> dict:
    """Run quality checks using precomputed KNN graph"""
    # Set fixed seeds for reproducibility
    np.random.seed(42)
    
    print("\n=== Embedding Quality Checks ===")
    
    # Convert distances to similarities (cosine distance = 1 - cosine similarity)
    similarities = 1 - distances
    
    # Calculate metrics directly from in-memory graph
    avg_similarities = similarities.mean(axis=1)
    similarity_variances = similarities.var(axis=1)
    
    # Global statistics
    global_mean = avg_similarities.mean()
    global_std = avg_similarities.std()
    var_mean = similarity_variances.mean()
    var_std = similarity_variances.std()
    
    # Outlier detection (2σ thresholds)
    outlier_mask = (
        (avg_similarities < (global_mean - 2 * global_std)) |
        (similarity_variances > (var_mean + 2 * var_std))
    )
    outlier_count = np.sum(outlier_mask)
    
    # Distribution analysis
    hist, bins = np.histogram(avg_similarities, bins=20, density=True)
    
    print("\nNeighborhood Coherence Metrics:")
    print(f"Global mean similarity: {global_mean:.3f} ± {global_std:.3f}")
    print(f"Variance stats: {var_mean:.3f} ± {var_std:.3f}")
    print(f"Outlier papers: {outlier_count} ({outlier_count/len(avg_similarities)*100:.1f}%)")
    print(f"Similarity range: [{avg_similarities.min():.3f}, {avg_similarities.max():.3f}]")
    
    # Plot distribution
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(avg_similarities, bins=20, alpha=0.7)
    plt.title('Distribution of Neighborhood Similarities')
    plt.xlabel('Average Similarity Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    return {
        'global_mean': global_mean,
        'global_std': global_std,
        'var_mean': var_mean,
        'var_std': var_std,
        'outlier_count': outlier_count,
        'similarity_distribution': (hist, bins)
    }

validation_results = validate_embeddings(distances, indices)

# %% [markdown]
# ## 7. Backup

# %%
def backup_embeddings():
    """Use pg_dump for PostgreSQL backups"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"
    !pg_dump -U postgres -F c -f "{backup_path}" papers # pyright: ignore

# Call backup after processing
backup_embeddings()

