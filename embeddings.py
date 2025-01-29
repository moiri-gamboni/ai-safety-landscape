# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # AI Safety Papers - Abstract Embedding Phase
# 
# This notebook generates Voyage AI embeddings for paper abstracts and stores them in the database.

# %% [markdown]
# ## 1. Environment Setup

# %%
# Mount Google Drive
from google.colab import drive # pyright: ignore [reportMissingImports]
drive.mount('/content/drive')

# Install required packages if running in Colab
import os
if 'COLAB_GPU' in os.environ:
    # Install only the packages needed for this notebook
    %pip install voyageai tqdm scikit-learn tenacity # pyright: ignore

# %% [markdown]
# ## 2. Database Connection

# %%
import sqlite3
import os
import numpy as np
from tqdm import tqdm
import voyageai
from sklearn.metrics.pairwise import cosine_similarity
import time
from typing import List, Tuple, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import re

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
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"
local_db = "papers.db"

# Copy database to local storage if needed
print(f"Copying database to local storage: {local_db}")
if not os.path.exists(local_db):
    %cp "{db_path}" {local_db} # pyright: ignore

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
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_abstract_embedding 
        ON papers(abstract_embedding)
    ''')
    conn.commit()

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
def embed_with_backoff(abstracts, model="voyage-3-large"):
    return vo.embed(
        abstracts,
        model=model,
        input_type="document",
        output_dimension=2048
    )

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

def process_batch(batch: List[Tuple[str, str]], model: str = "voyage-3-large") -> List[Tuple[str, List[float]]]:
    """Process a batch of papers, returning embeddings"""
    abstracts = [item[1] for item in batch]
    paper_ids = [item[0] for item in batch]
    
    # First count tokens for rate limiting
    try:
        token_count = vo.count_tokens(abstracts, model=model)
    except Exception as e:
        print(f"Error counting tokens: {str(e)}")
        raise
    
    # Check rate limits
    if not rate_limiter.can_make_request(token_count):
        raise Exception("Rate limit exceeded")  # This will trigger exponential backoff
    
    # Generate embeddings with exponential backoff
    try:
        result = embed_with_backoff(abstracts, model=model)
        
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
        print(f"First abstract: {abstracts[0][:500]}...")  # Truncate long abstracts
        raise

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
                        SET abstract_embedding = ?
                        WHERE id = ?
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
    
    # Compute similarities using dot product (faster for normalized vectors)
    n_samples = len(random_embeddings)
    if n_samples > 1:
        # Generate random pairs of indices
        idx1 = np.random.randint(0, n_samples, size=1000)
        idx2 = np.random.randint(0, n_samples, size=1000)
        # Exclude self-pairs
        valid_pairs = idx1 != idx2
        idx1, idx2 = idx1[valid_pairs], idx2[valid_pairs]
        
        # Compute dot products (equivalent to cosine similarity for normalized vectors)
        similarities = np.sum(random_embeddings[idx1] * random_embeddings[idx2], axis=1)
        
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
    
    # 5. Embedding Analysis
    print("\n5. Embedding Analysis:")
    
    def find_outliers(embeddings: np.ndarray, papers: list, n_examples: int = 3):
        """Find outlier papers based on their similarity to other papers
        
        Args:
            embeddings: Array of embeddings (each normalized to unit length)
            papers: List of paper records corresponding to embeddings
            n_examples: Number of outlier examples to show
        """
        # Calculate pairwise similarities
        similarities = cosine_similarity(embeddings)
        
        # For each paper, get average similarity to other papers
        # Exclude self-similarity (1.0) from the average
        avg_similarities = []
        for i in range(len(similarities)):
            others = np.concatenate([similarities[i,:i], similarities[i,i+1:]])
            avg_similarities.append(np.mean(others))
        
        avg_similarities = np.array(avg_similarities)
        
        # Use IQR method on average similarities
        q1, q3 = np.percentile(avg_similarities, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr  # Papers with unusually low average similarity
        
        # Find outliers (papers with unusually low average similarity to others)
        outlier_indices = np.where(avg_similarities < lower_bound)[0]
        
        print(f"\nOutlier Analysis:")
        print(f"- Average similarity stats:")
        print(f"  Mean: {np.mean(avg_similarities):.3f} ± {np.std(avg_similarities):.3f}")
        print(f"  Range: [{np.min(avg_similarities):.3f}, {np.max(avg_similarities):.3f}]")
        print(f"- Found {len(outlier_indices)} outliers ({len(outlier_indices)/len(embeddings)*100:.1f}%)")
        
        if len(outlier_indices) > 0:
            # Sort outliers by average similarity (ascending)
            sorted_outliers = sorted(zip(outlier_indices, avg_similarities[outlier_indices]), 
                                   key=lambda x: x[1])
            
            print(f"\nTop {n_examples} outliers (lowest average similarity to other papers):")
            for idx, avg_sim in sorted_outliers[:n_examples]:
                paper = papers[idx]
                print(f"\nAverage similarity to other papers: {avg_sim:.3f}")
                print(f"Title: {paper['title']}")
                print(f"URL: https://arxiv.org/abs/{paper['id']}")
                print(f"Abstract:\n{paper['abstract'][:500]}...")
                
                # Show a few most and least similar papers to this one
                sims = similarities[idx]
                most_similar_idx = np.argsort(sims)[-2]  # -1 would be self
                least_similar_idx = np.argsort(sims)[0]
                
                print(f"\nMost similar paper (similarity: {sims[most_similar_idx]:.3f}):")
                print(f"Title: {papers[most_similar_idx]['title']}")
                
                print(f"\nLeast similar paper (similarity: {sims[least_similar_idx]:.3f}):")
                print(f"Title: {papers[least_similar_idx]['title']}")
    
    # Get sample of embeddings
    cursor.execute('''
        SELECT id, title, abstract, abstract_embedding
        FROM papers 
        WHERE abstract_embedding IS NOT NULL
        LIMIT 1000
    ''')
    papers = cursor.fetchall()
    embeddings = [np.frombuffer(row['abstract_embedding'], dtype=np.float32) for row in papers]
    all_embeddings = np.array(embeddings)
    
    # Basic embedding stats
    print(f"- Shape: {all_embeddings.shape}")
    print("- Component statistics:")
    means = np.mean(all_embeddings, axis=0)
    stds = np.std(all_embeddings, axis=0)
    print(f"  Mean of means: {np.mean(means):.3f} ± {np.std(means):.3f}")
    print(f"  Mean of stds: {np.mean(stds):.3f} ± {np.std(stds):.3f}")
    print(f"  Component range: [{np.min(all_embeddings):.3f}, {np.max(all_embeddings):.3f}]")
    
    # Show distribution of components
    positive_frac = np.mean(all_embeddings > 0)
    print(f"  Fraction of positive components: {positive_frac:.3f}")
    
    # Find outliers using the new method
    find_outliers(all_embeddings, papers)

# Run validation
validate_embeddings(conn)

# %% [markdown]
# ## 5. Duplicate Analysis

# %%
import re
def find_duplicates(conn, similarity_threshold=0.95, output_file='duplicates.csv'):
    """Find potential duplicate papers based on title and embedding similarity
    
    Args:
        conn: Database connection
        similarity_threshold: Minimum similarity to consider as duplicate
        output_file: Path to save results (CSV)
    """
    print("Finding potential duplicates...")
    
    def normalize_title(title):
        """Normalize title for comparison"""
        if not title:
            return ""
        title = title.lower()
        title = re.sub(r'\bv\d+\b', '', title)
        title = re.sub(r'arxiv:\d+\.\d+', '', title, flags=re.IGNORECASE)
        title = re.sub(r'[^\w\s]', '', title)
        return ' '.join(title.split())
    
    # Get papers with embeddings
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, title, abstract_embedding
        FROM papers
        WHERE abstract_embedding IS NOT NULL
          AND withdrawn = 0
    ''')
    
    # Group by normalized title
    title_groups = {}
    for row in cursor:
        norm_title = normalize_title(row['title'])
        if norm_title:  # Skip empty titles
            if norm_title in title_groups:
                title_groups[norm_title].append(row)
            else:
                title_groups[norm_title] = [row]
    
    # Find duplicates based on both title and embedding similarity
    duplicates = []
    
    print(f"Analyzing {len(title_groups)} unique normalized titles...")
    for title, group in title_groups.items():
        if len(group) > 1:  # Only process groups with multiple papers
            # Compare embeddings within group
            embeddings = np.vstack([np.frombuffer(p['abstract_embedding'], dtype=np.float32) for p in group])
            similarities = cosine_similarity(embeddings)
            
            # Find pairs above threshold (excluding self-similarity)
            for i in range(len(similarities)):
                for j in range(i+1, len(similarities)):
                    if similarities[i,j] > similarity_threshold:
                        duplicates.append({
                            'id1': group[i]['id'],
                            'id2': group[j]['id'],
                            'title': group[i]['title'],
                            'similarity': float(similarities[i,j])
                        })
    
    # Sort by similarity
    duplicates.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Save results
    import csv
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id1', 'id2', 'title', 'similarity'])
        writer.writeheader()
        writer.writerows(duplicates)
    
    # Print summary
    print(f"\nFound {len(duplicates)} potential duplicate pairs")
    print(f"Results saved to {output_file}")
    
    # Print top examples
    if duplicates:
        print("\nTop 5 potential duplicates:")
        for dup in duplicates[:5]:
            print(f"\nSimilarity: {dup['similarity']:.3f}")
            print(f"Title: {dup['title']}")
            print(f"Paper 1: https://arxiv.org/abs/{dup['id1']}")
            print(f"Paper 2: https://arxiv.org/abs/{dup['id2']}")

# Run duplicate analysis
find_duplicates(conn)

# %% [markdown]
# ## 6. Save Results

# %%
# Save duplicates to Drive
%cp duplicates.csv "/content/drive/MyDrive/ai-safety-papers/duplicates.csv" # pyright: ignore
print(f"Duplicates saved to Drive")

# Copy updated database back to Drive
%cp {local_db} "{db_path}" # pyright: ignore
print("Database backup completed to Google Drive")

