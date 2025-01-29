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
    %pip install numpy optuna permetrics hdbscan umap-learn # pyright: ignore
    !git clone https://github.com/rapidsai/rapidsai-csp-utils.git # pyright: ignore
    !python rapidsai-csp-utils/colab/pip-install.py # pyright: ignore

# Core imports
import sqlite3
import numpy as np

# ML imports
from cuml import UMAP
from cuml.preprocessing import StandardScaler
from cuml.cluster.hdbscan import HDBSCAN
from cuml.metrics.trustworthiness import trustworthiness

# Optimization imports
import optuna
from permetrics import ClusteringMetric

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
    persistence REAL,
    PRIMARY KEY (run_id, parent_cluster_id, child_cluster_id),
    FOREIGN KEY (run_id) REFERENCES clustering_runs(run_id)
)''')

conn.commit()

# %% [markdown]
# ## 3. Data Loading

# %%
def load_embeddings():
    """Load and standardize embeddings once"""
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
    
    # Standardize once during initial load
    scaler = StandardScaler()
    raw_embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in results])
    scaled_embeddings = scaler.fit_transform(raw_embeddings)
    
    print(f"Loaded {len(paper_ids)} papers with standardized embeddings")
    return paper_ids, scaled_embeddings

paper_ids, embeddings = load_embeddings()

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

def perform_umap_reduction(embeddings, n_components, n_neighbors, min_dist):
    """Use pre-scaled embeddings, only apply UMAP if needed"""
    if n_components == 0:
        print("Using pre-standardized embeddings without reduction")
        return embeddings
    
    # Add detailed parameter print
    print(f"\nPerforming {n_components}D UMAP reduction with parameters:")
    print(f"n_neighbors: {n_neighbors}, min_dist: {min_dist}")
    
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        verbose=True
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
            ''', (run_id, pid, emb.astype(np.float32).tobytes()))
        
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
    embeddings = np.array([np.frombuffer(row[1], dtype=np.float32) for row in results])
    
    return embeddings, paper_ids

# %% [markdown]
# ## 5. Optimization Setup

# %%
def objective(trial, embeddings):
    """Optuna optimization objective function"""
    # UMAP configuration
    use_umap = trial.suggest_categorical('use_umap', [True, False])
    umap_params = {
        'min_dist': 0.0,
        'n_components': 0,  # Default for no UMAP
        'n_neighbors': 0    # Unused
    }
    
    if use_umap:
        umap_params.update({
            'n_components': trial.suggest_int('n_components', 15, 100),
            'n_neighbors': trial.suggest_int('n_neighbors', 30, 100)
        })
    
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
        return cursor.fetchone()['dbcvi_score']
    
    # Perform clustering
    clusterer = HDBSCAN(
        **hdbscan_params,
        metric='euclidean',
        prediction_data=True,
        gen_min_span_tree=True
    )
    labels = clusterer.fit_predict(reduced_embeddings)
    
    print(f"\nClustering complete. Found {len(np.unique(labels))-1} clusters "
          f"({np.sum(labels == -1)} noise points)")
    
    # Calculate metrics
    if not use_umap:
        trust_score = 1.0  # Max score when using original embeddings
    else:
        trust_score = trustworthiness(embeddings, reduced_embeddings)
    
    cm = ClusteringMetric(X=reduced_embeddings, y_pred=labels)
    dbcvi_score = cm.DBCVI()
    
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
        'mean_persistence': np.mean(persistence),
        'std_persistence': np.std(persistence)
    }
    
    cluster_sizes = [np.sum(clusterer.labels_ == label) 
                    for label in np.unique(clusterer.labels_) if label != -1]
    
    # Require valid clusters
    if not cluster_sizes:
        raise ValueError("No clusters found - all points labeled as noise")
    
    stats.update({
        'mean_cluster_size': np.mean(cluster_sizes),
        'std_cluster_size': np.std(cluster_sizes),
        'cluster_size_ratio': (max(cluster_sizes) / min(cluster_sizes))
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
        np.sum(clusterer.labels_ == -1) / len(clusterer.labels_),
        len(np.unique(clusterer.labels_[clusterer.labels_ != -1])),
        hierarchy_stats['mean_persistence'],
        hierarchy_stats['std_persistence'],
        hierarchy_stats.get('mean_cluster_size', 0),
        hierarchy_stats.get('std_cluster_size', 0),
        hierarchy_stats.get('cluster_size_ratio', 0)
    ))
    
    run_id = cursor.lastrowid
    conn.commit()
    print(f"Saved clustering run {run_id} with {hierarchy_stats['n_clusters']} clusters")
    return run_id

def save_cluster_hierarchy(run_id, condensed_tree):
    """Save cluster hierarchy data"""
    print("\nSaving cluster hierarchy relationships...")
    cursor = conn.cursor()
    count = 0
    for row in condensed_tree.itertuples():
        if row.child_size > 1:
            count += 1
            cursor.execute('''
                INSERT INTO cluster_hierarchy
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                run_id,
                int(row.parent),
                int(row.child),
                float(row.lambda_val),
                int(row.child_size),
                float(row.persistence)
            ))
    conn.commit()
    print(f"Saved {count} hierarchy relationships for run {run_id}")

def optimize_clustering(embeddings, n_trials=100):
    """Run optimization study"""
    study = optuna.create_study(direction='minimize')
    
    # Track best run across all trials
    best_score = float('inf')
    best_run_id = None
    
    def log_and_update_best(study, trial):
        """Enhanced callback with continuous best run tracking"""
        nonlocal best_score, best_run_id
        
        print(f"\nTrial {trial.number} finished:")
        print(f"Params: {trial.params}")
        print(f"Value: {trial.value:.3f}")
        
        # Update best run if improved
        current_best = study.best_trial
        if current_best.value < best_score:
            best_score = current_best.value
            new_best_id = current_best.user_attrs['db_run_id']
            
            if new_best_id != best_run_id:
                print(f"New best run found! Updating marker to run {new_best_id}")
                cursor = conn.cursor()
                # Clear previous optimal marker
                cursor.execute('UPDATE clustering_runs SET is_optimal = 0')
                # Set new optimal marker
                cursor.execute('''
                    UPDATE clustering_runs 
                    SET is_optimal = 1 
                    WHERE run_id = ?
                ''', (new_best_id,))
                conn.commit()
                best_run_id = new_best_id
    
    study.optimize(
        lambda trial: objective(trial, embeddings),
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
study = optimize_clustering(embeddings, n_trials=100)
print("Optimization complete! Best parameters saved to database.")

# %% [markdown]
# ## 7. Database Backup

# %%
# Copy updated database back to Drive
print("\nStarting database backup to Google Drive...")
%cp {local_db} "{db_path}" # pyright: ignore
print("Backup completed successfully")


