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
    %pip install optuna hdbscan umap-learn numpy cupy-cuda12x # pyright: ignore
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

# Locale fix after install https://github.com/googlecolab/colabtools/issues/3409
import locale
locale.getpreferredencoding = lambda: "UTF-8"

# Add to Core imports
from cuml.neighbors import NearestNeighbors

# %% [markdown]
# ## 2. Database Setup

# %%
# Database configuration
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"
local_db = "papers.db"

# Initialize database connection
print(f"Copying database to local storage: {local_db}")
if not os.path.exists(local_db):
    %cp "{db_path}" {local_db} # pyright: ignore

conn = sqlite3.connect(local_db)
conn.row_factory = sqlite3.Row

# Create tables
cursor = conn.cursor()

# UMAP tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS umap_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    n_components INTEGER,
    n_neighbors INTEGER,
    min_dist REAL
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS umap_results (
    run_id INTEGER,
    paper_id TEXT,
    embedding BLOB,
    PRIMARY KEY (run_id, paper_id),
    FOREIGN KEY (run_id) REFERENCES umap_runs(run_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
)''')

# Clustering tables
cursor.execute('''
CREATE TABLE IF NOT EXISTS clustering_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    umap_run_id INTEGER,
    is_optimal BOOLEAN DEFAULT 0,
    min_cluster_size INTEGER,
    min_samples INTEGER,
    cluster_selection_method TEXT,
    cluster_selection_epsilon REAL,
    trust_score REAL,
    dbcvi_score REAL,
    noise_ratio REAL,
    n_clusters INTEGER,
    mean_persistence REAL,
    std_persistence REAL,
    mean_cluster_size REAL,
    std_cluster_size REAL,
    cluster_size_ratio REAL,
    FOREIGN KEY (umap_run_id) REFERENCES umap_runs(run_id)
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS clustering_results (
    run_id INTEGER,
    paper_id TEXT,
    cluster_id INTEGER,
    cluster_prob REAL,
    PRIMARY KEY (run_id, paper_id),
    FOREIGN KEY (run_id) REFERENCES clustering_runs(run_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
)''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS cluster_hierarchy (
    run_id INTEGER,
    parent_cluster_id INTEGER,
    child_cluster_id INTEGER,
    lambda_val REAL,
    child_size INTEGER,
    PRIMARY KEY (run_id, parent_cluster_id, child_cluster_id),
    FOREIGN KEY (run_id) REFERENCES clustering_runs(run_id)
)''')

conn.commit()

# %% [markdown]
# ## 3. Data Loading

# %%
def load_embeddings():
    """Load embeddings and precompute k-NN graph"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, abstract_embedding 
        FROM papers 
        WHERE abstract_embedding IS NOT NULL AND withdrawn = 0
    ''')
    
    results = cursor.fetchall()
    if not results:
        raise ValueError("No embeddings found in database")
    
    # Initialize arrays
    paper_ids = [row[0] for row in results]
    
    # Pure cupy buffer conversion
    raw_embeddings = cp.array([cp.frombuffer(row[1], dtype=cp.float32) for row in results])
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(raw_embeddings)
    
    # Precompute k-NN graph with max neighbors needed
    print("Precomputing k-NN graph for UMAP...")
    nn_model = NearestNeighbors(n_neighbors=100, metric='cosine')
    nn_model.fit(scaled_embeddings)
    knn_graph = nn_model.kneighbors_graph(scaled_embeddings, mode='distance')
    
    print(f"Precomputed {knn_graph.shape[1]} neighbors for each sample")
    return paper_ids, scaled_embeddings, knn_graph

paper_ids, embeddings, knn_graph = load_embeddings()

# %% [markdown]
# ## 4. Core Functions

# %%
def check_existing_umap_run(n_components, n_neighbors, min_dist):
    """Check for existing UMAP run with matching parameters"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT run_id FROM umap_runs
        WHERE n_components = ? AND n_neighbors = ? AND min_dist = ?
    ''', (n_components, n_neighbors, min_dist))
    result = cursor.fetchone()
    return result['run_id'] if result else None

def perform_umap_reduction(embeddings, n_components, n_neighbors, min_dist, knn_graph):
    """UMAP using precomputed k-NN graph"""
    print(f"\nPerforming {n_components}D UMAP reduction with parameters:")
    print(f"n_neighbors: {n_neighbors}, min_dist: {min_dist}")
    
    # Validate neighbor count
    if n_neighbors > knn_graph.shape[1]:
        raise ValueError(f"Requested {n_neighbors} neighbors exceeds precomputed {knn_graph.shape[1]}")
    
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

def save_umap_run(paper_ids, embeddings, n_components, n_neighbors, min_dist):
    """Save UMAP results to database"""
    print(f"\nSaving UMAP results to database (n={len(paper_ids)})...")
    cursor = conn.cursor()
    cursor.execute('BEGIN')
    
    try:
        cursor.execute('''
            INSERT INTO umap_runs (n_components, n_neighbors, min_dist)
            VALUES (?, ?, ?)
        ''', (n_components, n_neighbors, min_dist))
        run_id = cursor.lastrowid
        
        for pid, emb in zip(paper_ids, embeddings):
            cursor.execute('''
                INSERT INTO umap_results (run_id, paper_id, embedding)
                VALUES (?, ?, ?)
            ''', (run_id, pid, emb.astype(cp.float32).get().tobytes()))
        
        conn.commit()
        print(f"Saved UMAP run {run_id} with {len(paper_ids)} entries")
        return run_id
    except Exception as e:
        conn.rollback()
        print(f"Failed to save UMAP run: {e}")
        return None

def load_umap_embeddings(run_id):
    """Load UMAP embeddings from database for a given run"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT paper_id, embedding 
        FROM umap_results 
        WHERE run_id = ?
        ORDER BY paper_id
    ''', (run_id,))
    
    results = cursor.fetchall()
    if not results:
        raise ValueError(f"No embeddings found for run {run_id}")
    
    paper_ids = [row[0] for row in results]
    embeddings = cp.array([cp.frombuffer(row[1], dtype=cp.float32) for row in results])
    
    return embeddings, paper_ids

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
    return float(np.sum(weighted_V))

def objective(trial, embeddings, knn_graph):
    """Optuna optimization objective function"""
    # UMAP configuration
    use_umap = trial.suggest_categorical('use_umap', [True, False])
    existing_umap_id = None
    reduced_embeddings = embeddings
    
    if use_umap:
        umap_params = {
            'n_components': trial.suggest_int('n_components', 15, 100),
            'n_neighbors': trial.suggest_int('n_neighbors', 30, 100),
            'min_dist': 0.0,
            'knn_graph': knn_graph
        }
        
        # Check for existing UMAP run
        existing_umap_id = check_existing_umap_run(**umap_params)
        
        if existing_umap_id:
            reduced_embeddings, _ = load_umap_embeddings(existing_umap_id)
        else:
            # Perform and save new UMAP reduction
            reduced_embeddings = perform_umap_reduction(embeddings, **umap_params)
            existing_umap_id = save_umap_run(paper_ids, reduced_embeddings, **umap_params)

    # HDBSCAN parameters
    min_cluster_size = trial.suggest_int('min_cluster_size', 20, 100)
    hdbscan_params = {
        'min_cluster_size': min_cluster_size,
        'min_samples': trial.suggest_int('min_samples', 5, min_cluster_size//2),
        'cluster_selection_method': trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf']),
        'cluster_selection_epsilon': trial.suggest_float('cluster_selection_epsilon', 0.0, 0.5)
    }
    
    # Add HDBSCAN parameter print
    print("\nUsing HDBSCAN parameters:")
    print(f"min_cluster_size: {hdbscan_params['min_cluster_size']}")
    print(f"min_samples: {hdbscan_params['min_samples']}")
    print(f"cluster_selection_method: {hdbscan_params['cluster_selection_method']}")
    print(f"cluster_selection_epsilon: {hdbscan_params['cluster_selection_epsilon']}")
    
    # Check for existing clustering run
    existing_cluster_id = check_existing_clustering_run(
        umap_run_id=existing_umap_id,
        **hdbscan_params
    )
    
    if existing_cluster_id:
        cursor = conn.cursor()
        cursor.execute('SELECT dbcvi_score FROM clustering_runs WHERE run_id = ?', (existing_cluster_id,))
        score = cursor.fetchone()['dbcvi_score']
        
        # Add this critical line to propagate the existing run ID
        trial.set_user_attr('db_run_id', existing_cluster_id)
        return score
    
    # Perform clustering
    clusterer = HDBSCAN(
        **hdbscan_params,
        metric='euclidean',
        prediction_data=True,
        gen_min_span_tree=True,
        output_type='cupy'
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    
    print(f"\nClustering complete. Found {len(cp.unique(labels))-1} clusters "
          f"({cp.sum(labels == -1).item()} noise points)")
    
    # Calculate metrics
    print("\nCalculating trustworthiness...")
    if not use_umap:
        trust_score = None
    else:
        trust_score = trustworthiness(embeddings, reduced_embeddings)
    
    print("\nCalculating DBCVI score...")
    dbcvi_score = compute_relative_validity(clusterer.minimum_spanning_tree_, labels)
    
    # Save results to DB
    run_id = save_optimized_run(
        existing_umap_id,
        hdbscan_params,
        clusterer,
        trust_score,
        dbcvi_score
    )
    
    # Store hierarchy and create visualization embedding
    save_cluster_hierarchy(run_id, clusterer.condensed_tree_)
    
    # Only create visualization embedding if we used UMAP
    if use_umap:
        create_visualization_embedding(umap_params, existing_umap_id)
    
    trial.set_user_attr('db_run_id', run_id)  # Store actual DB ID
    return dbcvi_score

def create_visualization_embedding(umap_params, main_run_id):
    """Ensure 2D visualization embedding exists"""
    if umap_params['n_components'] == 2:
        return
    
    viz_params = umap_params.copy()
    viz_params['n_components'] = 2
    viz_run_id = check_existing_umap_run(**viz_params)
    
    if viz_run_id:
        print(f"Using existing 2D visualization embedding (run {viz_run_id})")
    else:
        print("\nCreating new 2D visualization embedding...")
        viz_embeddings = perform_umap_reduction(embeddings, **viz_params)
        viz_run_id = save_umap_run(paper_ids, viz_embeddings, **viz_params)
        print(f"Created 2D visualization embedding (run {viz_run_id})")

def check_existing_clustering_run(umap_run_id, **hdbscan_params):
    """Check for existing clustering run with these parameters"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT run_id FROM clustering_runs
        WHERE umap_run_id = ?
        AND min_cluster_size = ?
        AND min_samples = ?
        AND cluster_selection_method = ?
        AND cluster_selection_epsilon = ?
    ''', (
        umap_run_id,
        hdbscan_params['min_cluster_size'],
        hdbscan_params['min_samples'],
        hdbscan_params['cluster_selection_method'],
        hdbscan_params['cluster_selection_epsilon']
    ))
    result = cursor.fetchone()
    return result['run_id'] if result else None

def analyze_hierarchy(clusterer):
    """Remove error suppression for persistence metrics"""
    persistence = clusterer.cluster_persistence_
    stats = {
        'mean_persistence': cp.mean(persistence),
        'std_persistence': cp.std(persistence)
    }
    
    # Get valid labels (exclude noise)
    labels = clusterer.labels_
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    
    if valid_labels.size == 0:
        raise ValueError("No clusters found - all points labeled as noise")
    
    # Calculate cluster sizes using bincount
    cluster_sizes = cp.bincount(valid_labels)
    
    stats.update({
        'mean_cluster_size': cluster_sizes.mean(),
        'std_cluster_size': cluster_sizes.std(),
        'cluster_size_ratio': (cluster_sizes.max() / cluster_sizes.min())
    })
    
    return stats

def save_optimized_run(umap_run_id, hdbscan_params, clusterer, trust_score, dbcvi_score):
    """Save optimized clustering results to database"""
    print("\nSaving clustering metrics to database...")
    cursor = conn.cursor()
    hierarchy_stats = analyze_hierarchy(clusterer)
    
    cursor.execute('''
        INSERT INTO clustering_runs (
            umap_run_id, min_cluster_size, min_samples, cluster_selection_method, cluster_selection_epsilon,
            trust_score, dbcvi_score, noise_ratio, n_clusters,
            mean_persistence, std_persistence, mean_cluster_size, std_cluster_size, cluster_size_ratio
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        umap_run_id,
        hdbscan_params['min_cluster_size'],
        hdbscan_params['min_samples'],
        hdbscan_params['cluster_selection_method'],
        hdbscan_params['cluster_selection_epsilon'],
        trust_score,
        dbcvi_score,
        cp.sum(clusterer.labels_ == -1).item() / len(clusterer.labels_),
        len(cp.unique(clusterer.labels_[clusterer.labels_ != -1])),
        hierarchy_stats['mean_persistence'].item(),
        hierarchy_stats['std_persistence'].item(),
        hierarchy_stats.get('mean_cluster_size', 0).item(),
        hierarchy_stats.get('std_cluster_size', 0).item(),
        hierarchy_stats.get('cluster_size_ratio', 0).item()
    ))
    
    run_id = cursor.lastrowid
    conn.commit()
    print(f"Saved clustering run {run_id} with {len(cp.unique(clusterer.labels_[clusterer.labels_ != -1]))} clusters")
    return run_id

def save_cluster_hierarchy(run_id, condensed_tree):
    """Save cluster hierarchy relationships (CPU-based)"""
    print("\nSaving cluster hierarchy relationships...")
    cursor = conn.cursor()
    
    # Convert condensed tree to pandas DataFrame
    tree_df = condensed_tree.to_pandas()
    
    # Filter meaningful relationships (exclude single-point clusters)
    meaningful_edges = tree_df[tree_df.child_size > 1]

    # Batch insert using executemany with correct columns
    cursor.executemany('''
        INSERT INTO cluster_hierarchy
        VALUES (?, ?, ?, ?, ?)
    ''', [
        (run_id, int(row.parent), int(row.child), 
         float(row.lambda_val), int(row.child_size))
        for row in meaningful_edges.itertuples()
    ])
    
    conn.commit()
    print(f"Saved {len(meaningful_edges)} hierarchy relationships for run {run_id}")

def optimize_clustering(embeddings, knn_graph, n_trials=50):
    """Run optimization study"""
    study = optuna.create_study(direction='maximize')
    
    # Track best run across all trials
    best_score = -float('inf')
    best_run_id = None
    
    def log_and_update_best(study, trial):
        """Enhanced callback with continuous best run tracking"""
        nonlocal best_score, best_run_id
        
        print(f"\nTrial {trial.number} finished:")
        print(f"Params: {trial.params}")
        print(f"Value: {trial.value:.3f}")
        
        # Update best run if improved
        current_best = study.best_trial
        if current_best.value > best_score:
            best_score = current_best.value
            new_best_id = current_best.user_attrs['db_run_id']
            
            if new_best_id != best_run_id:
                print(f"New best run found! Updating marker to run {new_best_id}")
                cursor = conn.cursor()
                cursor.execute('UPDATE clustering_runs SET is_optimal = 0')
                cursor.execute('''
                    UPDATE clustering_runs 
                    SET is_optimal = 1 
                    WHERE run_id = ?
                ''', (new_best_id,))
                conn.commit()
                best_run_id = new_best_id
    
    study.optimize(
        lambda trial: objective(trial, embeddings, knn_graph),
        n_trials=n_trials,
        callbacks=[log_and_update_best]  # Use enhanced callback
    )
    
    print(f"\nFinal best trial ({study.best_trial.number}):")
    print(f"Score: {study.best_trial.value:.3f}")
    print("Parameters:", study.best_trial.params)
    
    return study

# %% [markdown]
# ## 6. Run Optimization

# %%
study = optimize_clustering(embeddings, knn_graph, n_trials=50)
print("Optimization complete! Best parameters saved to database.")

# %% [markdown]
# ## 7. Database Backup

# %%
# Copy updated database back to Drive
print("\nStarting database backup to Google Drive...")
%cp {local_db} "{db_path}" # pyright: ignore
print("Backup completed successfully")


