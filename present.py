# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# ## 1. Setup

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
    
    %pip install joblib psycopg2-binary optuna hdbscan umap-learn numpy cupy-cuda12x # pyright: ignore
    !git clone https://github.com/rapidsai/rapidsai-csp-utils.git # pyright: ignore
    !python rapidsai-csp-utils/colab/pip-install.py # pyright: ignore

# Core imports
import sqlite3
import cupy as cp
import numpy as np

# ML imports
from cuml import UMAP
from cuml.preprocessing import StandardScaler
from cuml.cluster.hdbscan import HDBSCAN
from cuml.metrics.trustworthiness import trustworthiness
import cuml
cuml.set_global_output_type('cupy')

# Optimization imports
import optuna
from optuna.trial import TrialState

# Locale fix after install https://github.com/googlecolab/colabtools/issues/3409
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Add to Core imports
from cuml.neighbors import NearestNeighbors

# Additional imports
from tenacity import retry, stop_after_attempt, wait_random_exponential
from google import genai
from google.genai import types
import time
import asyncio
from tqdm.auto import tqdm
import json

# Add after imports but before database setup
# %% [markdown]
# ## LLM Setup

# %%
# @title Gemini API Key
gemini_api_key = "" # @param {type:"string"}
MODEL_ID = "gemini-1.5-flash"

# Initialize Gemini client
client = genai.Client(api_key=gemini_api_key)

# Add rate limiter from labeling.py
class GeminiRateLimiter:
    def __init__(self):
        self.rpm_limit = 2000
        self.tpm_limit = 4_000_000
        self.requests = []
        self.tokens = []
        
    def can_make_request(self, input_tokens, output_tokens):
        current_time = time.time()
        cutoff_time = current_time - 60
        
        recent_requests = [t for t in self.requests if t > cutoff_time]
        if len(recent_requests) >= self.rpm_limit:
            delay = 60 - (current_time - cutoff_time)
            print(f"RPM limit reached. Retrying after {delay:.1f}s")
            time.sleep(delay)
            return False
            
        recent_tokens = sum(c for t, c in self.tokens if t > cutoff_time)
        total_tokens = recent_tokens + input_tokens + output_tokens
        if total_tokens > self.tpm_limit:
            print(f"TPM limit exceeded ({total_tokens}/{self.tpm_limit})")
            return False
            
        return True
        
    def add_request(self, input_tokens, output_tokens):
        current_time = time.time()
        self.requests.append(current_time)
        self.tokens.append((current_time, input_tokens + output_tokens))
        time.sleep(0.05)

rate_limiter = GeminiRateLimiter()

# %% [markdown]
# ## 2. Database Setup

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
    
    %pip install joblib psycopg2-binary optuna hdbscan umap-learn numpy cupy-cuda12x # pyright: ignore
    !git clone https://github.com/rapidsai/rapidsai-csp-utils.git # pyright: ignore
    !python rapidsai-csp-utils/colab/pip-install.py # pyright: ignore

# Core imports
import cupy as cp
import numpy as np

# ML imports
from cuml import UMAP
from cuml.preprocessing import StandardScaler
from cuml.cluster.hdbscan import HDBSCAN
from cuml.metrics.trustworthiness import trustworthiness
import cuml

# Optimization imports
import optuna
from optuna.trial import TrialState

# Locale fix after install https://github.com/googlecolab/colabtools/issues/3409
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Add to Core imports
from cuml.neighbors import NearestNeighbors

# Additional imports
import pickle
from itertools import islice
import gc
from joblib import parallel_backend
# %%
backup_path = "/content/drive/MyDrive/ai-safety-papers/filtered_compressed.db"

def get_db_connection():
    """Create PostgreSQL connection with retries"""
    import psycopg2
    from psycopg2.extras import DictCursor
    
    return psycopg2.connect(
        host='',  # Empty string for Unix socket connection
        database="papers",
        user="postgres",
        cursor_factory=DictCursor
    )

# After creating connection but before creating tables:
print("Loading existing database...")
!createdb -U postgres papers # pyright: ignore
!pg_restore -U postgres --jobs=8 -d papers "{backup_path}" # pyright: ignore
conn = get_db_connection()

import json 
# Load best trial immediately after connection
def get_best_trial():
    """Load best trial ID and metrics from JSON"""
    drive_path = "/content/drive/MyDrive/ai-safety-papers/best_trial.json"
    with open(drive_path) as f:
        return json.load(f)

best_trial_data = get_best_trial()
best_trial = best_trial_data['trial_id']


# %%
def get_best_clusterer():
    """Reconstruct clusterer using stored embeddings and params from JSON"""
    # Get parameters and embeddings
    cluster_params = best_trial_data['params']
    
    # Load UMAP reduced embeddings from database
    paper_ids = []
    umap_embeddings = []
    with conn.cursor() as cursor:
        cursor.execute('''
            SELECT paper_id, umap_embedding
            FROM artifacts
            WHERE trial_id = %s AND umap_embedding IS NOT NULL
            ORDER BY paper_id
        ''', (best_trial,))
        
        for paper_id, emb_bytes in cursor:
            paper_ids.append(paper_id)
            # Convert bytea back to cupy array
            umap_embeddings.append(cp.frombuffer(emb_bytes, dtype=cp.float32))
    
    # Create embeddings matrix
    reduced_embeddings = cp.stack(umap_embeddings)
    
    # Reconstruct and fit clusterer with original parameters
    clusterer = HDBSCAN(
        min_cluster_size=cluster_params['min_cluster_size'],
        min_samples=cluster_params['min_samples'],
        cluster_selection_epsilon=cluster_params['cluster_selection_epsilon'],
        cluster_selection_method='leaf',
        gen_min_span_tree=True,
        output_type='cupy'
    ).fit(reduced_embeddings)
    
    # Create paper_id to index mapping
    paper_id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}
    
    return clusterer, paper_ids, reduced_embeddings, paper_id_to_idx

# Load clusterer and results after defining best_trial
best_clusterer, paper_ids, reduced_embeddings, paper_id_to_idx = get_best_clusterer()

# %% [markdown]
# ## Leaf Cluster Labeling

# %%
def get_cluster_members(cluster_id):
    """Retrieve papers using clusterer labels array"""
    # Get indices where label matches cluster_id
    mask = best_clusterer.labels_.get() == cluster_id
    member_ids = [paper_ids[i] for i in np.where(mask)[0]]
    
    # Fetch details from database
    with conn.cursor() as cursor:
        cursor.execute('''
            SELECT id, title, abstract 
            FROM papers 
            WHERE id = ANY(%s)
        ''', (member_ids,))
        return cursor.fetchall()

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
async def generate_cluster_label_async(representatives, cluster_id):
    """Generate label and safety relevance score for a cluster"""
    # Build paper list string
    papers_str = "\n\n".join(
        f"Title: {p['title']}\nAbstract: {p['abstract']}..."  # Truncate long abstracts
        for p in representatives[:10]  # Use first 10 as most representative
    )
    
    prompt = f"""You are an expert in AI safety and machine learning. Your task is to generate precise technical labels for clusters of academic papers related to AI research.

I will provide the ten papers most representative of the cluster (closest to the cluster centroid).

Review these papers and provide:
1. A specific technical category that precisely describes the research area represented by this cluster
2. A relevance score (0-1) indicating how relevant this research area is to AI safety

Guidelines:
- Use precise technical terminology
- Categories should be specific enough to differentiate between related research areas yet broad enough to actually group papers (e.g. "Reward Modeling for RLHF" rather than "Reinforcement Learning" or "Regularizing Hidden States Enables Learning Generalizable Reward Model for RLHF")
- Consider both direct and indirect relevance to AI safety

Papers to analyze:
{papers_str}"""
    
    # Define response schema
    schema = {
        "type": "OBJECT",
        "properties": {
            "label": {"type": "STRING"},
            "safety_relevance": {
                "type": "NUMBER",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": ["label", "safety_relevance"]
    }
    
    response = await client.aio.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema
        )
    )
    
    # Update rate limiter with actual usage
    input_tokens = response.usage_metadata.prompt_token_count
    output_tokens = response.usage_metadata.candidates_token_count
    rate_limiter.add_request(input_tokens, output_tokens)
    
    return json.loads(response.text)

async def process_leaf_clusters_async():
    """Async version of leaf cluster processing"""
    semaphore = asyncio.Semaphore(20)
    labels = best_clusterer.labels_.get()
    leaf_clusters = [cid for cid in np.unique(labels) if cid != -1]
    
    # Precompute centroids
    centroids = get_cluster_centroids()
    
    async def process_cluster(cluster_id):
        async with semaphore:
            members = await asyncio.to_thread(get_cluster_members, cluster_id)
            if len(members) < 5:
                return
                
            # Get correct embeddings using the mapping
            member_ids = [m['id'] for m in members]
            indices = [paper_id_to_idx[pid] for pid in member_ids]
            cluster_embeddings = reduced_embeddings[indices]
            
            # Find papers closest to centroid
            centroid = centroids[cluster_id]
            distances = cp.linalg.norm(cluster_embeddings - centroid, axis=1)
            
            # Sort members by distance
            sorted_indices = cp.argsort(distances).get().tolist()
            sorted_members = [members[i] for i in sorted_indices]
            
            # Select representatives
            representatives = [
                *sorted_members[:10]  # Top 10 closest to centroid
            ]
            
            # Get label and safety relevance
            label_data = await generate_cluster_label_async(representatives, cluster_id)
            await asyncio.to_thread(update_cluster_label, cluster_id, label_data, representatives)

    tasks = [process_cluster(cid) for cid in leaf_clusters]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Labeling clusters"):
        await f

def update_cluster_label(cluster_id, label_data, representatives):
    """Store label and relevance score in database"""
    with conn.cursor() as cursor:
        cursor.execute('''
            INSERT INTO cluster_labels
            (cluster_id, label, safety_relevance, representative_ids)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (cluster_id) DO UPDATE SET
                label = EXCLUDED.label,
                safety_relevance = EXCLUDED.safety_relevance,
                representative_ids = EXCLUDED.representative_ids
        ''', (
            cluster_id,
            label_data['label'],
            label_data['safety_relevance'],
            [r['id'] for r in representatives]
        ))
        conn.commit()

# Add this before process_leaf_clusters()
def create_label_columns():
    """Create columns for cluster labels"""
    with conn.cursor() as cursor:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cluster_labels (
                cluster_id INTEGER PRIMARY KEY,
                label TEXT,
                safety_relevance REAL,
                representative_ids TEXT[],
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()

def get_cluster_centroids():
    """Calculate centroids for each cluster using reduced embeddings"""
    centroids = {}
    labels = best_clusterer.labels_.get()
    unique_clusters = np.unique(labels[labels != -1])
    
    for cid in unique_clusters:
        cluster_mask = labels == cid
        cluster_embeddings = reduced_embeddings[cluster_mask]
        centroids[cid] = cluster_embeddings.mean(axis=0)
    
    return centroids

create_label_columns()

# %%
await process_leaf_clusters_async()

# %%

# %% [markdown]
# ## Test LLM Label Generation

# %%
async def test_single_cluster_labeling():
    """Test label generation for a single cluster"""
    try:
        # Get first non-noise cluster
        labels = best_clusterer.labels_.get()
        valid_clusters = [cid for cid in np.unique(labels) if cid != -1]
        if not valid_clusters:
            print("No clusters available for testing")
            return
            
        test_cluster = valid_clusters[0]
        print(f"Testing label generation for cluster {test_cluster}")
        
        # Get members
        members = await asyncio.to_thread(get_cluster_members, test_cluster)
        if len(members) < 5:
            print("Cluster too small for testing")
            return
            
        # Get representatives
        member_ids = [m['id'] for m in members]
        indices = [paper_id_to_idx[pid] for pid in member_ids]
        cluster_embeddings = reduced_embeddings[indices]
        centroid = reduced_embeddings[best_clusterer.labels_.get() == test_cluster].mean(axis=0)
        distances = cp.linalg.norm(cluster_embeddings - centroid, axis=1)
        sorted_indices = cp.argsort(distances).get().tolist()
        representatives = [members[i] for i in sorted_indices[:10]]
        
        # Generate label
        print("\nSample papers:")
        for p in representatives[:2]:  # Show first 2 for verification
            print(f"\nTitle: {p['title']}")
            print(f"Abstract: {p['abstract'][:200]}...")
            
        print("\nGenerated label:")
        label_data = await generate_cluster_label_async(representatives, test_cluster)
        print(json.dumps(label_data, indent=2))
        
    except Exception as e:
        print(f"Test failed: {str(e)}")

# Run test before full processing
await test_single_cluster_labeling()
