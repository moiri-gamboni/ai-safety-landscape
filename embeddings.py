# %% [markdown]
# # AI Safety Papers - Abstract Embedding Phase
# 
# This notebook generates ModernBERT embeddings for paper abstracts and stores them in the database.

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

# Verify GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 2. Database Connection

# %%
import sqlite3
import os
import numpy as np
from tqdm import tqdm

# Path to database
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"
local_db = "papers.db"

# Copy database to local storage if needed
print(f"Copying database to local storage: {local_db}")
if not os.path.exists(local_db):
    %cp "{db_path}" {local_db}

conn = sqlite3.connect(local_db)
conn.row_factory = sqlite3.Row

# Check if abstract_embedding column exists
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

# %% [markdown]
# ## 3. Model Initialization

# %%
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import torch.nn.functional as F

MODEL_NAME = "answerdotai/ModernBERT-large"

# Load model and tokenizer (only once)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    attn_implementation="eager"  # Use standard attention implementation for broader compatibility
).to(device)
model.eval()

# %% [markdown]
# ## 4. Embedding Generation

# %%
def generate_embeddings(
    texts: List[str],
    pooling: str = "mean",
    normalize: bool = True
) -> np.ndarray:
    """Generate embeddings for a batch of texts using ModernBERT.
    
    Args:
        texts: List of texts to embed
        pooling: Pooling strategy ('mean' or 'cls')
            - mean: Average all token embeddings (better for retrieval/clustering)
            - cls: Use [CLS] token embedding
        normalize: Whether to L2-normalize embeddings (recommended for clustering)
    
    Returns:
        numpy.ndarray: Array of embeddings, shape (n_texts, 1024)
    """
    # Tokenize texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,  # Use ModernBERT's full context length
        return_tensors="pt"
    ).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state
        
        if pooling == "mean":
            # Mean pooling with attention mask
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        else:  # cls pooling
            embeddings = hidden_states[:, 0]
        
        # Optionally normalize
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy().astype(np.float32)

def get_csai_papers(conn: sqlite3.Connection, batch_size_ref: dict):
    """Generator that yields batches of papers with cs.AI in their categories
    
    Args:
        conn: Database connection
        batch_size_ref: Dictionary containing current batch size, allows for dynamic updates
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
            if len(batch) >= batch_size_ref['size']:
                pbar.update(len(batch))
                yield batch
                batch = []
                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if batch:
            pbar.update(len(batch))
            yield batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def get_gpu_memory_free():
    """Get actual free GPU memory in GB, accounting for all reserved memory"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        return total - reserved
    return 0

def get_gpu_memory_stats():
    """Get GPU memory statistics in GB"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        reserved = torch.cuda.max_memory_reserved() / 1024**3
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        return {
            'total': total,
            'reserved': reserved,
            'allocated': allocated,
            'cached': cached,
            'free_total': total - reserved,
            'free_cached': cached - allocated
        }
    return None

def calculate_batch_adjustment(current_size: int, peak_memory: float) -> Optional[int]:
    """Calculate new batch size based on memory usage.
    
    Args:
        current_size: Current batch size
        peak_memory: Peak memory used by current batch in GB
        
    Returns:
        Optional[int]: New batch size if adjustment needed, None otherwise
    """
    stats = get_gpu_memory_stats()
    if not stats:
        return None
        
    # How many times our current batch could fit in memory (leaving 2GB buffer)
    memory_headroom = (stats['total'] - 2.0) / peak_memory if peak_memory > 0 else 1
    
    # Target range: 1.2-1.4x headroom
    if memory_headroom > 1.4:  # Using too little memory
        # Increase batch size to target 1.3x headroom
        increase_factor = min(memory_headroom / 1.3, 1.3)
        return int(current_size * increase_factor)
    elif memory_headroom < 1.2:  # Using too much memory
        # Reduce batch size to target 1.3x headroom
        decrease_factor = 1.3 / memory_headroom
        return int(current_size / decrease_factor)
    return None  # We're in the optimal range

# Process in batches
total_updated = 0
batch_size_ref = {'size': 256}  # Mutable reference to batch size

try:
    while True:  # Keep trying until we succeed or hit an unrecoverable error
        try:
            for batch in get_csai_papers(conn, batch_size_ref):
                # Reset peak stats before processing
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                paper_ids = [item[0] for item in batch]
                abstracts = [item[1] for item in batch]
                
                # Generate embeddings with mean pooling and normalization
                embeddings = generate_embeddings(
                    abstracts,
                    pooling="mean",  # Better for document similarity
                    normalize=True   # Better for clustering
                )
                
                # Calculate actual memory used by this batch
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                
                # Store in database
                cursor = conn.cursor()
                for paper_id, embedding in zip(paper_ids, embeddings):
                    embedding_blob = embedding.tobytes()
                    cursor.execute('''
                        UPDATE papers
                        SET abstract_embedding = ?
                        WHERE id = ?
                    ''', (embedding_blob, paper_id))
                
                conn.commit()
                total_updated += len(batch)
                
                # Check if we should adjust batch size
                new_size = calculate_batch_adjustment(batch_size_ref['size'], peak_memory)
                if new_size:
                    old_size = batch_size_ref['size']
                    batch_size_ref['size'] = new_size
                    print(f"\nAdjusting batch size: {old_size} → {new_size} (peak memory: {peak_memory:.1f}GB)")
                
                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            
            # If we complete the loop without OOM errors, we're done
            break
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                old_size = batch_size_ref['size']
                # Reduce by 20%
                batch_size_ref['size'] = int(old_size * 0.8)
                print(f"\nCUDA out of memory - reducing batch size: {old_size} → {batch_size_ref['size']}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                continue
            raise  # Re-raise if it's not an OOM error

except Exception as e:
    print(f"\nFatal error in embedding generation: {str(e)}")
    raise
finally:
    print(f"\nProcess completed:")
    print(f"- Total papers processed: {total_updated}")

# %% [markdown]
# ## 5. Data Quality Validation

# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def validate_embeddings(conn):
    """Run quality checks on generated embeddings"""
    # Set fixed seeds for reproducibility
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
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
# ## 6. Database Backup

# %%
# Copy updated database back to Drive
!cp {local_db} "{db_path}"
print("Database backup completed to Google Drive")
