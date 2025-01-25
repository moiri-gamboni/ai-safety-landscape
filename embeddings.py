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
    torch_dtype=torch.bfloat16
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

def get_csai_papers(conn: sqlite3.Connection, batch_size: int = 256):
    """Generator that yields batches of CS.AI papers with abstracts"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, abstract 
        FROM papers 
        WHERE categories LIKE 'cs.AI%' 
          AND abstract IS NOT NULL
          AND abstract_embedding IS NULL
    ''')
    
    batch = []
    for row in cursor:
        batch.append((row['id'], row['abstract']))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

# %%
# Process in batches
total_updated = 0
batch_size = 256  # Adjust based on GPU memory

for batch in tqdm(get_csai_papers(conn, batch_size), desc="Processing batches"):
    paper_ids = [item[0] for item in batch]
    abstracts = [item[1] for item in batch]
    
    try:
        # Generate embeddings with mean pooling and normalization
        embeddings = generate_embeddings(
            abstracts,
            pooling="mean",  # Better for document similarity
            normalize=True   # Better for clustering
        )
        
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
        print(f"Processed {total_updated} papers", end='\r')
        
    except Exception as e:
        print(f"\nError processing batch: {str(e)}")
        conn.rollback()

print(f"\nCompleted! Total papers updated: {total_updated}")

# %% [markdown]
# ## 5. Database Backup

# %%
# Copy updated database back to Drive
!cp {local_db} "{db_path}"
print("Database backup completed to Google Drive")

# %% [markdown]
# ## 6. Data Quality Validation

# %%
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def validate_embeddings(conn):
    """Run quality checks on generated embeddings"""
    cursor = conn.cursor()
    
    print("\n=== Embedding Quality Checks ===")
    
    # 1. Coverage Check
    cursor.execute('''
        SELECT 
            COUNT(*) AS total_csai,
            SUM(CASE WHEN abstract_embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding,
            SUM(CASE WHEN abstract_embedding IS NULL THEN 1 ELSE 0 END) AS without_embedding
        FROM papers 
        WHERE categories LIKE 'cs.AI%'
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
        LIMIT 1000
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
        ORDER BY RANDOM()
        LIMIT 1000
    ''')
    norms = []
    for row in cursor:
        embedding = np.frombuffer(row['abstract_embedding'], dtype=np.float32)
        norms.append(np.linalg.norm(embedding))
    
    print(f"\n3. Norm Analysis (sample):")
    print(f"- Mean norm: {np.mean(norms):.2f}")
    print(f"- Std dev: {np.std(norms):.2f}")
    print(f"- Min/Max: {np.min(norms):.2f}/{np.max(norms):.2f}")

    # 4. Similarity Sanity Check
    # Get random pairs
    cursor.execute('''
        SELECT abstract_embedding 
        FROM papers 
        WHERE abstract_embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 100
    ''')
    random_embeddings = [np.frombuffer(row['abstract_embedding'], dtype=np.float32) for row in cursor]
    random_similarities = cosine_similarity(random_embeddings)
    np.fill_diagonal(random_similarities, np.nan)  # Exclude self-similarity
    
    # Get duplicate candidates (from harvesting phase)
    cursor.execute('''
        SELECT p1.abstract_embedding, p2.abstract_embedding
        FROM (
            SELECT id, title
            FROM papers
            WHERE withdrawn = 0
        ) p1
        JOIN (
            SELECT id, title
            FROM papers
            WHERE withdrawn = 0
        ) p2 ON p1.title COLLATE NOCASE = p2.title COLLATE NOCASE AND p1.id < p2.id
        WHERE p1.abstract_embedding IS NOT NULL
          AND p2.abstract_embedding IS NOT NULL
        LIMIT 50
    ''')
    duplicate_pairs = []
    for row in cursor:
        e1 = np.frombuffer(row[0], dtype=np.float32)
        e2 = np.frombuffer(row[1], dtype=np.float32)
        duplicate_pairs.append((e1, e2))
    
    # Calculate similarities
    duplicate_similarities = [cosine_similarity([e1], [e2])[0][0] for e1, e2 in duplicate_pairs]
    
    print(f"\n4. Similarity Analysis:")
    print(f"- Random pairs (n={len(random_similarities)**2 - len(random_similarities)}):")
    print(f"  Mean: {np.nanmean(random_similarities):.2f} ± {np.nanstd(random_similarities):.2f}")
    if duplicate_pairs:
        print(f"- Duplicate candidates (n={len(duplicate_pairs)}):")
        print(f"  Mean: {np.mean(duplicate_similarities):.2f} ± {np.std(duplicate_similarities):.2f}")
    else:
        print("- No duplicate pairs found for comparison")

# Run validation
validate_embeddings(conn)