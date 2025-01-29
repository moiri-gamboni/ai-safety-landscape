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
    # Install only the packages needed for this notebook
    %pip install --extra-index-url=https://pypi.nvidia.com numpy scikit-learn cuml-cu12==24.12.* tqdm # pyright: ignore

# Core imports
import sqlite3
import numpy as np
from tqdm import tqdm

# ML imports - fail fast if GPU versions aren't available
from cuml import UMAP  # GPU-accelerated UMAP
from cuml.preprocessing import StandardScaler
from cuml.cluster.hdbscan import HDBSCAN  # GPU-accelerated HDBSCAN
from cuml.metrics.cluster.silhouette_score import cython_silhouette_score as silhouette_score
from cuml.metrics.trustworthiness import trustworthiness
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from scipy.spatial.distance import cdist

# %% [markdown]
# ## 2. Database Setup

# %%
# Path to database
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"
local_db = "papers.db"

# Copy database to local storage if needed
print(f"Copying database to local storage: {local_db}")
if not os.path.exists(local_db):
    %cp "{db_path}" {local_db} # pyright: ignore

conn = sqlite3.connect(local_db)
conn.row_factory = sqlite3.Row

# Create UMAP runs table
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS umap_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- UMAP parameters
    n_components INTEGER,
    n_neighbors INTEGER,
    min_dist REAL
)
''')

# Create UMAP results table
cursor.execute('''
CREATE TABLE IF NOT EXISTS umap_results (
    run_id INTEGER,
    paper_id TEXT,
    embedding BLOB,
    PRIMARY KEY (run_id, paper_id),
    FOREIGN KEY (run_id) REFERENCES umap_runs(run_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
)
''')

# Create clustering runs table
cursor.execute('''
CREATE TABLE IF NOT EXISTS clustering_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    umap_run_id INTEGER,
    
    -- HDBSCAN parameters
    min_cluster_size INTEGER,
    min_samples INTEGER,
    cluster_selection_method TEXT,
    
    -- Metrics
    n_clusters INTEGER,
    noise_ratio REAL,
    silhouette_score REAL,
    davies_bouldin_score REAL,
    calinski_harabasz_score REAL,
    avg_coherence REAL,
    avg_separation REAL,
    
    FOREIGN KEY (umap_run_id) REFERENCES umap_runs(run_id)
)
''')

# Create clustering results table
cursor.execute('''
CREATE TABLE IF NOT EXISTS clustering_results (
    run_id INTEGER,
    paper_id TEXT,
    cluster_id INTEGER,
    cluster_prob REAL,
    PRIMARY KEY (run_id, paper_id),
    FOREIGN KEY (run_id) REFERENCES clustering_runs(run_id),
    FOREIGN KEY (paper_id) REFERENCES papers(id)
)
''')

# Create hierarchy table
cursor.execute('''
CREATE TABLE IF NOT EXISTS cluster_hierarchy (
    run_id INTEGER,
    parent_cluster_id INTEGER,
    child_cluster_id INTEGER,
    lambda_val REAL,           -- Split level in hierarchy
    child_size INTEGER,        -- Number of papers in child cluster
    PRIMARY KEY (run_id, parent_cluster_id, child_cluster_id),
    FOREIGN KEY (run_id) REFERENCES clustering_runs(run_id)
)
''')

conn.commit()

# %% [markdown]
# ## 3. Load Data

# %%
def load_embeddings():
    """Load paper embeddings from database"""
    cursor = conn.cursor()
    
    print("Loading papers with embeddings...")
    
    # First get the total count
    cursor.execute('''
        SELECT COUNT(*) as count
        FROM papers
        WHERE abstract_embedding IS NOT NULL
          AND withdrawn = 0
    ''')
    total_count = cursor.fetchone()['count']
    
    # Now get all embeddings
    cursor.execute('''
        SELECT id, abstract_embedding
        FROM papers
        WHERE abstract_embedding IS NOT NULL
          AND withdrawn = 0
    ''')
    
    # Get first row to determine embedding dimension
    first_row = cursor.fetchone()
    first_embedding = np.frombuffer(first_row['abstract_embedding'], dtype=np.float32)
    embedding_dim = len(first_embedding)
    
    # Pre-allocate arrays
    papers = [first_row['id']]
    embeddings = np.empty((total_count, embedding_dim), dtype=np.float32)
    embeddings[0] = first_embedding  # Store first embedding
    
    # Load the rest
    for i, row in enumerate(tqdm(cursor, total=total_count-1), start=1):
        papers.append(row['id'])
        embeddings[i] = np.frombuffer(row['abstract_embedding'], dtype=np.float32)
    
    print(f"\nLoaded {len(papers)} papers with embeddings")
    print(f"Embedding shape: {embeddings.shape}")
    
    return papers, embeddings

# Load the data
paper_ids, embeddings = load_embeddings()

# %% [markdown]
# ## 4. UMAP Reduction

# %% 
# @title UMAP Parameters {"run": "auto"}
n_components = 4  # @param {type:"slider", min:2, max:8, step:1}
n_neighbors = 15  # @param {type:"slider", min:5, max:50, step:5}
min_dist = 0.05  # @param {type:"slider", min:0.01, max:0.5, step:0.05}

# %%
def check_existing_umap_run(n_components, n_neighbors, min_dist):
    """Check if a UMAP run with these parameters exists"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT run_id 
        FROM umap_runs 
        WHERE n_components = ? 
          AND n_neighbors = ? 
          AND min_dist = ?
    ''', (n_components, n_neighbors, min_dist))
    
    result = cursor.fetchone()
    return result['run_id'] if result else None

def perform_umap_reduction(embeddings, n_components, n_neighbors, min_dist):
    """Reduce dimensionality using UMAP"""
    print(f"Performing {n_components}D UMAP reduction...")
    
    # Scale the embeddings
    scaler = StandardScaler(
        with_mean=True,  # Center the data
        with_std=True,   # Scale to unit variance
        copy=True        # Create a copy to avoid modifying original data
    )
    scaled_embeddings = scaler.fit_transform(embeddings.astype(np.float32))
    
    # UMAP reduction with cuML-specific parameters
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        init='spectral',
        random_state=42,  # For reproducibility
        n_epochs=None,    # Auto-select based on dataset size
        negative_sample_rate=5,
        transform_queue_size=4.0,
        verbose=True
    )
    
    reduced = reducer.fit_transform(scaled_embeddings)
    print(f"Generated {n_components}D embeddings")
    
    return reduced

def save_umap_run(paper_ids, embeddings, n_components, n_neighbors, min_dist):
    """Save UMAP run to database"""
    cursor = conn.cursor()
    
    # Start transaction
    cursor.execute('BEGIN')
    
    try:
        # Save run parameters
        cursor.execute('''
            INSERT INTO umap_runs (
                n_components,
                n_neighbors,
                min_dist
            ) VALUES (?, ?, ?)
        ''', (n_components, n_neighbors, min_dist))
        
        run_id = cursor.lastrowid
        
        # Save embeddings
        for i, paper_id in enumerate(tqdm(paper_ids, desc="Saving embeddings")):
            # Store as float32 to match the original embeddings
            embedding_bytes = embeddings[i].astype(np.float32).tobytes()
            cursor.execute('''
                INSERT INTO umap_results (
                    run_id,
                    paper_id,
                    embedding
                ) VALUES (?, ?, ?)
            ''', (run_id, paper_id, embedding_bytes))
        
        conn.commit()
        print(f"Saved UMAP run {run_id}")
        return run_id
    except:
        # If anything fails, rollback the transaction
        conn.rollback()
        print("Failed to save UMAP run, rolling back changes")
        return None

# Check for existing run with requested dimensions
cluster_run_id = check_existing_umap_run(n_components, n_neighbors, min_dist)

if cluster_run_id:
    print(f"Found existing {n_components}D UMAP run: {cluster_run_id}")
else:
    # Perform and save nD reduction
    reduced_nd = perform_umap_reduction(embeddings, n_components, n_neighbors, min_dist)
    cluster_run_id = save_umap_run(paper_ids, reduced_nd, n_components, n_neighbors, min_dist)

# Check if we need a separate 2D run for visualization
if n_components != 2:
    viz_run_id = check_existing_umap_run(2, n_neighbors, min_dist)
    
    if viz_run_id:
        print(f"Found existing 2D UMAP run: {viz_run_id}")
    else:
        # Perform and save 2D reduction with same parameters
        reduced_2d = perform_umap_reduction(embeddings, 2, n_neighbors, min_dist)
        viz_run_id = save_umap_run(paper_ids, reduced_2d, 2, n_neighbors, min_dist)
        
else:
    viz_run_id = cluster_run_id

# %% [markdown]
# ### UMAP Validation

# %%
# Load the saved UMAP run for validation
cursor = conn.cursor()
cursor.execute('''
    SELECT paper_id, embedding
    FROM umap_results
    WHERE run_id = ?
    ORDER BY paper_id
''', (cluster_run_id,))

results = cursor.fetchall()
reduced_embeddings = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in results])

# Check for NaN/Inf values
if np.any(np.isnan(reduced_embeddings)) or np.any(np.isinf(reduced_embeddings)):
    print("WARNING: UMAP produced NaN or Inf values!")

# Check for degenerate cases (all points same/too close)
distances = cdist(reduced_embeddings, reduced_embeddings)
np.fill_diagonal(distances, np.inf)  # Ignore self-distances
min_distances = np.min(distances, axis=1)

# Calculate trustworthiness score
trust_score = trustworthiness(
    embeddings,
    reduced_embeddings,
    n_neighbors=n_neighbors,
    metric='euclidean',
    convert_dtype=True
)

validation_metrics = {
    'min_distance': float(np.min(min_distances)),
    'max_distance': float(np.max(distances)),
    'mean_distance': float(np.mean(distances)),
    'std_distance': float(np.std(distances)),
    'trustworthiness': float(trust_score)
}

if validation_metrics['min_distance'] < 1e-10:
    print("WARNING: Some points are extremely close together!")
if validation_metrics['std_distance'] < 1e-6:
    print("WARNING: Points might be too uniformly distributed!")

print("\nUMAP Validation Metrics:")
print(f"- Min distance between points: {validation_metrics['min_distance']:.3e}")
print(f"- Max distance between points: {validation_metrics['max_distance']:.3f}")
print(f"- Mean distance between points: {validation_metrics['mean_distance']:.3f}")
print(f"- Std of distances: {validation_metrics['std_distance']:.3f}")
print(f"- Trustworthiness score: {validation_metrics['trustworthiness']:.3f}")

# %% [markdown]
# ## 5. HDBSCAN Clustering

# %%
# @title HDBSCAN Parameters {"run": "auto"}
min_cluster_size = 20  # @param {type:"slider", min:5, max:100, step:5}
min_samples = 5  # @param {type:"slider", min:1, max:20, step:1}
cluster_selection_method = "eom"  # @param ["leaf", "eom"]

# %%
def check_existing_clustering_run(umap_run_id, min_cluster_size, min_samples, cluster_selection_method):
    """Check if a clustering run with these parameters exists"""
    cursor = conn.cursor()
    cursor.execute('''
        SELECT run_id 
        FROM clustering_runs 
        WHERE umap_run_id = ?
          AND min_cluster_size = ? 
          AND min_samples = ? 
          AND cluster_selection_method = ?
    ''', (umap_run_id, min_cluster_size, min_samples, cluster_selection_method))
    
    result = cursor.fetchone()
    return result['run_id'] if result else None

def load_umap_embeddings(run_id):
    """Load UMAP embeddings from database"""
    cursor = conn.cursor()
    
    # Get paper IDs and embeddings
    cursor.execute('''
        SELECT paper_id, embedding
        FROM umap_results
        WHERE run_id = ?
        ORDER BY paper_id
    ''', (run_id,))
    
    results = cursor.fetchall()
    # The embeddings are stored as float32 in the database
    embeddings = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in results])
    paper_ids = [r[0] for r in results]
    
    return embeddings, paper_ids

def evaluate_clustering(embeddings, reduced_embeddings, labels, probabilities):
    """Calculate comprehensive clustering quality metrics"""
    metrics = {}
    
    # Only evaluate assigned points (not noise)
    mask = labels != -1
    if np.sum(mask) > 1:
        # Standard clustering metrics using cuML's silhouette score
        metrics['silhouette'] = cython_silhouette_score(
            reduced_embeddings[mask],
            labels[mask],
            metric='euclidean',
            convert_dtype=True
        )
        
        metrics['davies_bouldin'] = davies_bouldin_score(
            reduced_embeddings[mask],
            labels[mask]
        )
        metrics['calinski_harabasz'] = calinski_harabasz_score(
            reduced_embeddings[mask],
            labels[mask]
        )
        
        # Cluster statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        metrics['n_clusters'] = n_clusters
        metrics['noise_ratio'] = n_noise / len(labels)
        metrics['avg_cluster_size'] = np.sum(mask) / n_clusters if n_clusters > 0 else 0
        metrics['avg_probability'] = np.mean(probabilities[mask])
        
        # Semantic coherence using reduced embeddings
        unique_labels = sorted(set(labels[mask]))
        coherence_scores = []
        separation_scores = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_embeddings = reduced_embeddings[cluster_mask]
            
            # Calculate average pairwise similarity within cluster
            similarities = 1 - cdist(cluster_embeddings, cluster_embeddings, metric='euclidean')
            np.fill_diagonal(similarities, 0)  # Exclude self-similarity
            coherence_scores.append(np.mean(similarities))

            # Calculate separation from other clusters
            other_embeddings = reduced_embeddings[~cluster_mask]
            if len(other_embeddings) > 0:
                between_similarities = 1 - cdist(cluster_embeddings, other_embeddings, metric='euclidean')
                separation_scores.append(np.mean(between_similarities))
                
        metrics['avg_coherence'] = np.mean(coherence_scores)
        metrics['avg_separation'] = np.mean(separation_scores) if separation_scores else 0

    else:
        # Default values for failed clustering
        metrics.update({
            'silhouette': 0,
            'davies_bouldin': float('inf'),
            'calinski_harabasz': 0,
            'n_clusters': 0,
            'noise_ratio': 1.0,
            'avg_cluster_size': 0,
            'avg_probability': 0,
            'avg_coherence': 0,
            'avg_separation': 0
        })
    
    return metrics

def save_clustering_run(umap_run_id, paper_ids, labels, probabilities, metrics, clusterer):
    """Save clustering results to database"""
    cursor = conn.cursor()
    
    # Start transaction
    cursor.execute('BEGIN')
    
    try:
        # Save run parameters and metrics
        cursor.execute('''
            INSERT INTO clustering_runs (
                umap_run_id,
                min_cluster_size,
                min_samples,
                cluster_selection_method,
                n_clusters,
                noise_ratio,
                silhouette_score,
                davies_bouldin_score,
                calinski_harabasz_score,
                avg_coherence,
                avg_separation
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            umap_run_id,
            min_cluster_size,
            min_samples,
            cluster_selection_method,
            metrics['n_clusters'],
            metrics['noise_ratio'],
            metrics['silhouette'],
            metrics['davies_bouldin'],
            metrics['calinski_harabasz'],
            metrics['avg_coherence'],
            metrics['avg_separation']
        ))
        
        run_id = cursor.lastrowid
        
        # Save hierarchical structure
        print("Saving hierarchical structure...")
        tree = clusterer.condensed_tree_
        for row in tqdm(tree.itertuples(), desc="Saving hierarchy"):
            if row.child_size > 1:  # Skip individual points
                cursor.execute('''
                    INSERT INTO cluster_hierarchy (
                        run_id,
                        parent_cluster_id,
                        child_cluster_id,
                        lambda_val,
                        child_size
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    run_id,
                    int(row.parent),
                    int(row.child),
                    float(row.lambda_val),
                    int(row.child_size)
                ))
        
        # Save paper results
        print("Saving paper assignments...")
        for i, paper_id in enumerate(tqdm(paper_ids, desc="Saving results")):
            cursor.execute('''
                INSERT INTO clustering_results (
                    run_id,
                    paper_id,
                    cluster_id,
                    cluster_prob
                ) VALUES (?, ?, ?, ?)
            ''', (
                run_id,
                paper_id,
                int(labels[i]),
                float(probabilities[i])
            ))
        
        # Commit all changes
        conn.commit()
        print(f"Saved clustering run {run_id}")
        return run_id
    except:
        # If anything fails, rollback the transaction
        conn.rollback()
        print("Failed to save clustering run, rolling back changes")
        return None

# Check for existing run
existing_run_id = check_existing_clustering_run(cluster_run_id, min_cluster_size, min_samples, cluster_selection_method)

if existing_run_id:
    print(f"Found existing clustering run: {existing_run_id}")
else:
    # Load UMAP embeddings
    reduced_embeddings, paper_ids = load_umap_embeddings(cluster_run_id)
    
    # Initialize clusterer
    reduced_embeddings = reduced_embeddings.astype(np.float32)
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method=cluster_selection_method,
        prediction_data=True,  # Enable prediction capabilities
        gen_min_span_tree=True,  # Required for visualization
        verbose=True
    )
    
    # Perform clustering
    print("Performing HDBSCAN clustering...")
    labels = clusterer.fit_predict(reduced_embeddings)
    probabilities = clusterer.probabilities_
    
    # Evaluate results
    metrics = evaluate_clustering(embeddings, reduced_embeddings, labels, probabilities)
    
    # Print clustering statistics
    print("\nClustering Results:")
    print(f"- Number of clusters: {metrics['n_clusters']}")
    print(f"- Noise points: {metrics['noise_ratio']*100:.1f}%")
    print(f"- Average cluster size: {metrics['avg_cluster_size']:.1f}")
    print(f"- Average cluster probability: {metrics['avg_probability']:.3f}")
    print(f"\nQuality Metrics:")
    print(f"- Silhouette Score: {metrics['silhouette']:.3f}")
    print(f"- Davies-Bouldin Index: {metrics['davies_bouldin']:.3f}")
    print(f"- Calinski-Harabasz Index: {metrics['calinski_harabasz']:.3f}")
    print(f"- Average Semantic Coherence: {metrics['avg_coherence']:.3f}")
    print(f"- Average Cluster Separation: {metrics['avg_separation']:.3f}")
    
    # Save results
    run_id = save_clustering_run(cluster_run_id, paper_ids, labels, probabilities, metrics, clusterer)

# %% [markdown]
# ## 6. Database Backup

# %%
# Copy updated database back to Drive
%cp {local_db} "{db_path}" # pyright: ignore
print("Database backup completed to Google Drive")