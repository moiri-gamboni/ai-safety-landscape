# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # AI Safety Papers - Clustering Phase
# 
# This notebook performs clustering analysis on the paper embeddings to identify AI Safety relevant clusters:
# 1. Loads paper embeddings from the database
# 2. Performs UMAP dimensionality reduction (stored separately for reuse)
# 3. Applies HDBSCAN clustering using stored UMAP embeddings
# 4. Evaluates cluster quality and stores results
#
# Note: For visualizations and analysis, see visualizations.py

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
import pickle
from itertools import islice
import gc
from joblib import parallel_backend

# %% [markdown]
# ## 2. Database Setup

# %%
# Database configuration
db_backup_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"

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

# %%
# Create tables
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS artifacts (
    trial_id INTEGER NOT NULL,
    paper_id TEXT NOT NULL,
    umap_embedding BYTEA,
    cluster_id INTEGER,
    cluster_prob REAL,
    PRIMARY KEY (trial_id, paper_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS cluster_trees (
    trial_id INTEGER NOT NULL,
    parent_cluster_id INTEGER,
    child_cluster_id INTEGER,
    lambda_val REAL,
    child_size INTEGER,
    PRIMARY KEY (trial_id, parent_cluster_id, child_cluster_id)
)''')

conn.commit()

# %% [markdown]
# ## 3. Data Loading

# %%
def load_embeddings():
    """Load embeddings and precompute k-NN graph"""
    with conn.cursor() as cursor:
        cursor.execute('''
            SELECT id, embedding 
            FROM papers 
            WHERE embedding IS NOT NULL AND withdrawn = FALSE
        ''')
        print(f"Loading embeddings")
        results = cursor.fetchall()
    if not results:
        raise ValueError("No embeddings found in database")
    
    # Initialize arrays
    paper_ids = [row[0] for row in results]
    
    print(f"Scaling embeddings")
    # Pure cupy buffer conversion
    raw_embeddings = cp.array([cp.frombuffer(row[1], dtype=cp.float32) for row in results])
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(raw_embeddings)
    del raw_embeddings, results  # Free original data
    
    # Precompute k-NN graph with max neighbors needed
    print("Precomputing k-NN graph for UMAP...")
    nn_model = NearestNeighbors(n_neighbors=100, metric='cosine')
    nn_model.fit(scaled_embeddings)
    knn_graph = nn_model.kneighbors_graph(scaled_embeddings, mode='distance')
    
    print(f"Done precomputing graph")
    return paper_ids, scaled_embeddings, knn_graph

paper_ids, embeddings, knn_graph = load_embeddings()

# %% [markdown]
# ## 4. Core Functions

# %%
def perform_umap_reduction(embeddings, n_components, n_neighbors, min_dist, knn_graph):
    """UMAP using precomputed k-NN graph"""
    print(f"\nPerforming {n_components}D UMAP reduction with parameters:")
    print(f"n_neighbors: {n_neighbors}, min_dist: {min_dist}")
    
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        precomputed_knn=knn_graph,
        metric='cosine',
        output_type='cupy'
    )
    result = reducer.fit_transform(embeddings)
    print(f"UMAP reduction complete. Output shape: {result.shape}")
    return result

# %% [markdown]
# ## 5. Optimization Setup

# %%
def compute_relative_validity(minimum_spanning_tree, labels):
    """CPU-based relative validity score using HDBSCAN's MST"""
    # Convert labels to numpy array for CPU operations
    labels = cp.asnumpy(labels)  # Move to CPU
    
    # Extract edge information from MST (already CPU-based)
    mst_df = minimum_spanning_tree.to_pandas()
    
    # Initialize metrics
    noise_mask = labels == -1
    valid_labels = labels[~noise_mask]
    
    if valid_labels.size == 0:
        return -1.0  # All noise case
    
    cluster_sizes = np.bincount(valid_labels)
    num_clusters = len(cluster_sizes)
    total = len(labels)
    
    # Use numpy instead of cupy
    DSC = np.zeros(num_clusters, dtype=np.float32)
    DSPC_wrt = np.ones(num_clusters, dtype=np.float32) * np.inf
    max_distance = 0.0
    min_outlier_sep = np.inf

    # Process edges using vectorized operations
    edge_data = mst_df[['from', 'to', 'distance']].values
    for from_idx, to_idx, length in edge_data:
        max_distance = max(max_distance, length)
        
        label1 = labels[int(from_idx)]
        label2 = labels[int(to_idx)]
        
        if label1 == -1 and label2 == -1:
            continue
        elif label1 == -1 or label2 == -1:
            min_outlier_sep = min(min_outlier_sep, length)
            continue
            
        if label1 == label2:
            DSC[label1] = max(length, DSC[label1])
        else:
            DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
            DSPC_wrt[label2] = min(length, DSPC_wrt[label2])

    # Handle edge cases
    if np.isinf(min_outlier_sep):
        min_outlier_sep = max_distance if num_clusters > 1 else max_distance
        
    # Correct infinite values
    correction = 2.0 * (max_distance if num_clusters > 1 else min_outlier_sep)
    DSPC_wrt = np.where(DSPC_wrt == np.inf, correction, DSPC_wrt)
    
    # Compute final score
    V_index = (DSPC_wrt - DSC) / np.maximum(DSPC_wrt, DSC)
    weighted_V = (cluster_sizes * V_index) / total
    result = float(np.sum(weighted_V))
    
    # Explicit cleanup
    del labels, mst_df
    return result

def get_optuna_storage():
    return "postgresql://postgres@/postgres"  # Omit host entirely for Unix socket

def save_sampler(study):
    """Save sampler state to Google Drive"""
    drive_path = "/content/drive/MyDrive/ai-safety-papers"
    sampler_path = f"{drive_path}/sampler.pkl"
    with open(sampler_path, "wb") as f:
        pickle.dump(study.sampler, f)

def load_sampler():
    """Load sampler from Google Drive if exists"""
    drive_path = "/content/drive/MyDrive/ai-safety-papers"
    sampler_path = f"{drive_path}/sampler.pkl"
    if os.path.exists(sampler_path):
        with open(sampler_path, "rb") as f:
            return pickle.load(f)
    return None

def calculate_metrics(cluster_persistences, labels, use_umap, original_embeddings, processed_embeddings):
    """Calculate all metrics while maintaining GPU arrays where possible"""
    metrics = {}
    
    if use_umap:
        # Keep data on GPU for trustworthiness calculation
        metrics['trust_score'] = trustworthiness(original_embeddings, processed_embeddings)
    else:
        metrics['trust_score'] = None
    
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    
    if valid_labels.size > 0:
        # Use Cupy for GPU-accelerated calculations
        cluster_sizes = cp.bincount(valid_labels)
        persistence = cluster_persistences
        
        metrics.update({
            'noise_ratio': cp.sum(~valid_mask).item() / len(labels),
            'n_clusters': len(cluster_sizes),
            'mean_persistence': cp.mean(persistence).item(),
            'std_persistence': cp.std(persistence).item(),
            'mean_cluster_size': cluster_sizes.mean().item(),
            'std_cluster_size': cluster_sizes.std().item(),
            'cluster_size_ratio': (cluster_sizes.max() / cluster_sizes.min()).item()
        })
    else:
        metrics.update({
            'noise_ratio': 1.0,
            'n_clusters': 0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'mean_cluster_size': 0.0,
            'std_cluster_size': 0.0,
            'cluster_size_ratio': 0.0
        })
    
    return metrics

BATCH_SIZE = 1000

def batched(iterable, n):
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch

def objective(trial, scaled_embeddings, knn_graph):
    """Optuna optimization objective function"""
    # UMAP configuration
    use_umap = trial.suggest_categorical('use_umap', [True, False])
    
    if use_umap:
        umap_params = {
            'n_components': trial.suggest_int('n_components', 15, 100),
            'n_neighbors': trial.suggest_int('n_neighbors', 30, 100),
            'min_dist': 0.0
        }
        
        # Always compute fresh UMAP
        reducer = UMAP(
            **umap_params,
            precomputed_knn=knn_graph,
            metric='cosine',
            output_type='cupy'
        )
        reduced_embeddings = reducer.fit_transform(scaled_embeddings).astype(cp.float32)
        del reducer
    else:
        reduced_embeddings = scaled_embeddings  # Already cupy

    # HDBSCAN parameters
    clusterer = HDBSCAN(
        min_cluster_size=trial.suggest_int('min_cluster_size', 20, 100),
        min_samples=trial.suggest_int('min_samples', 5, 50),
        cluster_selection_epsilon=trial.suggest_float('cluster_selection_epsilon', 0.0, 0.5),
        cluster_selection_method='leaf',
        gen_min_span_tree=True,
        output_type='cupy'
    )
    
    # Extract needed components first
    labels = clusterer.fit_predict(reduced_embeddings)
    mst = clusterer.minimum_spanning_tree_
    tree_df = clusterer.condensed_tree_.to_pandas()
    probabilities = clusterer.probabilities_
    cluster_persistences = clusterer.cluster_persistence_
    del clusterer  # ← Release hierarchy data

    # Calculate metrics
    metrics = calculate_metrics(cluster_persistences, labels, use_umap, scaled_embeddings, reduced_embeddings)
    dbcvi_score = compute_relative_validity(mst, labels)

    # Print metrics in readable format
    print(f"\nTrial {trial.number} results:")
    print(f"  • Noise ratio: {metrics['noise_ratio']:.1%}")
    print(f"  • Clusters: {metrics['n_clusters']}")
    print(f"  • Avg cluster size: {metrics['mean_cluster_size']:.1f} ± {metrics['std_cluster_size']:.1f}")
    print(f"  • Trust score: {metrics['trust_score']:.3f}" if metrics['trust_score'] else "  • Trust score: N/A")
    print(f"  • DBCVI score: {dbcvi_score:.3f}")

    # Store metrics (excluding dbcvi_score which is the objective value)
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    
    # Save combined artifacts
    for batch in batched((
        (trial.number, pid, 
         emb.tobytes() if use_umap else None,
         int(cluster.item()), 
         float(prob.item()))
        for pid, emb, cluster, prob in zip(
            paper_ids, 
            reduced_embeddings,
            labels.get(),  # Convert cupy→numpy once for entire array
            probabilities.get()  # Same here
        )
    ), BATCH_SIZE):
        cursor.executemany('''
            INSERT INTO artifacts
            VALUES (%s, %s, %s, %s, %s)
        ''', batch)
        conn.commit()
    
    # Save hierarchy tree
    meaningful_edges = tree_df[tree_df.child_size > 1]
    cursor.executemany('''
        INSERT INTO cluster_trees
        VALUES (%s, %s, %s, %s, %s)
    ''', [
        (trial.number, int(row.parent), int(row.child), 
         float(row.lambda_val), int(row.child_size))
        for row in meaningful_edges.itertuples()
    ])
    
    conn.commit()
    return dbcvi_score

def backup_database():
    """Backup PostgreSQL database to Google Drive"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"
    print(f"Creating PostgreSQL backup at {backup_path}")
    !pg_dump -U postgres -F c -f "{backup_path}" papers  # pyright: ignore
    print("Backup completed successfully")

# Combined backup callback that saves both sampler and database
def backup(study, _):
    if len(study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))) % 10 == 0:
        save_sampler(study)  # Save sampler state first
        backup_database()     # Then backup database

def optimize_clustering(embeddings, knn_graph, n_jobs, n_trials):
    """Run optimization study with Optuna integration"""
    study = optuna.create_study(
        study_name="ai-papers-clustering",
        storage=get_optuna_storage(),
        direction='maximize',
        load_if_exists=True,
        sampler=load_sampler()
    )
    
    with parallel_backend('multiprocessing'):
        study.optimize(
            lambda trial: objective(trial, embeddings, knn_graph),
            n_jobs=n_jobs,
            callbacks=[
                backup,
                optuna.study.MaxTrialsCallback(
                    n_trials, 
                    states=(TrialState.COMPLETE,)
                )
            ],
            gc_after_trial=True
        )
    
    return study

# %% [markdown]
# ## 6. Run Optimization

# %%
gc.collect()
study = optimize_clustering(embeddings, knn_graph, n_jobs=8, n_trials=500)
print("Optimization complete! Best parameters saved to database.")

# %% [markdown]
# ## 7. Database Backup

# %%
# Unassign GPU to free up resources
from google.colab import runtime # pyright: ignore [reportMissingImports]
runtime.unassign()

