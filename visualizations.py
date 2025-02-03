# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # AI Safety Papers - Clustering Visualizations
# 
# This notebook provides visualizations and analysis tools for exploring the clustering results:
# 1. Basic cluster visualization
# 2. Cluster metrics and statistics
# 3. Hierarchical structure exploration
# 4. Cluster content analysis

# %% [markdown]
# ## 1. Setup

#%%
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
    %pip install psycopg2-binary umap-learn # pyright: ignore
    %pip install matplotlib seaborn scipy scikit-learn networkx umap-learn hdbscan # pyright: ignore


# %% [markdown]
# ## 2. Database Connection

# %%
import psycopg2
import os
import numpy as np
from psycopg2.extras import DictCursor
import json

# Path to database
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"

def load_database():
    """Load PostgreSQL backup using psql"""
    print("Loading PostgreSQL backup...")
    !createdb -U postgres papers # pyright: ignore
    !pg_restore -U postgres --jobs=8 -d papers "{db_path}" # pyright: ignore

def connect_db():
    """Connect to PostgreSQL database with schema validation"""
    conn = psycopg2.connect(
        host='',
        database="papers",
        user="postgres",
        cursor_factory=DictCursor
    )
    return conn

load_database()
conn = connect_db()

# Load best trial immediately after connection
def get_best_trial():
    """Load best trial ID and metrics from JSON"""
    drive_path = "/content/drive/MyDrive/ai-safety-papers/best_trial.json"
    with open(drive_path) as f:
        return json.load(f)

best_trial_data = get_best_trial()
best_trial = best_trial_data['trial_id']

# %% [markdown]
# ## Imports and Configuration

# %%
# Core imports
import numpy as np
import json

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider
import networkx as nx
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist, squareform
from umap import UMAP

# %% [markdown]
# ## Helper Functions

# %%
def extract_cluster_keywords(papers, n_keywords=10):
    """Extract keywords using TF-IDF"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Combine title and abstract
    texts = [f"{p['title']} {p['abstract']}" for p in papers]
    
    # TF-IDF with custom parameters for scientific text
    vectorizer = TfidfVectorizer(
        max_df=0.7,                      # Ignore terms that appear in >70% of docs
        min_df=3,                        # Ignore terms that appear in <3 docs
        stop_words='english',
        ngram_range=(1, 2),             # Include bigrams
        token_pattern=r'(?u)\b[A-Za-z][A-Za-z-]+\b'  # Words starting with letter
    )
    
    # Get TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Sum TF-IDF scores for each term
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    
    # Get top keywords
    top_indices = scores.argsort()[-n_keywords:][::-1]
    keywords = [(feature_names[i], scores[i]) for i in top_indices]
    
    return keywords

def get_clusters_at_lambda(lambda_val):
    """Get cluster assignments at a specific lambda level"""
    with conn.cursor() as cursor:
        cursor.execute('''
            WITH RECURSIVE cluster_tree AS (
                SELECT ct.parent_cluster_id as cluster_id, ct.lambda_val
                FROM cluster_trees ct
                WHERE ct.trial_id = %s
                AND ct.parent_cluster_id NOT IN (
                    SELECT child_cluster_id 
                    FROM cluster_trees 
                    WHERE trial_id = %s
                )
                UNION ALL
                SELECT ct.child_cluster_id, ct.lambda_val
                FROM cluster_trees ct
                JOIN cluster_tree t ON ct.parent_cluster_id = t.cluster_id
                WHERE ct.trial_id = %s
                AND ct.lambda_val >= %s
            )
            SELECT DISTINCT cluster_id, lambda_val
            FROM cluster_tree
            WHERE lambda_val >= %s
            ORDER BY cluster_id
        ''', (best_trial, best_trial, best_trial, lambda_val, lambda_val))
        return cursor.fetchall()

def get_papers_in_cluster(cluster_id):
    """Get all papers belonging to a specific cluster"""
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT a.paper_id, p.title, p.abstract, a.cluster_prob
        FROM artifacts a
        JOIN papers p ON a.paper_id = p.id
        WHERE a.trial_id = %s AND a.cluster_id = %s
        ORDER BY a.cluster_prob DESC
    ''', (best_trial, cluster_id))
    
    return cursor.fetchall()

def get_cluster_hierarchy_levels():
    """Get all available lambda levels in the hierarchy"""
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT DISTINCT lambda_val
        FROM cluster_trees
        WHERE trial_id = %s
        ORDER BY lambda_val DESC
    ''', (best_trial,))
    
    return [row[0] for row in cursor.fetchall()]

def analyze_cluster_at_level(cluster_id, lambda_val):
    """Analyze a specific cluster at a given level"""
    cursor = conn.cursor()
    
    # Get papers in this cluster
    papers = get_papers_in_cluster(cluster_id)
    
    # Get immediate child clusters
    cursor.execute('''
        SELECT child_cluster_id, child_size
        FROM cluster_trees
        WHERE trial_id = %s 
          AND parent_cluster_id = %s
          AND lambda_val >= %s
        ORDER BY child_size DESC
    ''', (best_trial, cluster_id, lambda_val))
    
    children = cursor.fetchall()
    
    # Extract keywords for the cluster
    paper_dicts = [{'title': p[1], 'abstract': p[2]} for p in papers]
    keywords = extract_cluster_keywords(paper_dicts) if paper_dicts else []
    
    return {
        'cluster_id': cluster_id,
        'size': len(papers),
        'keywords': keywords,
        'children': children,
        'papers': papers
    }

def calculate_validity_metrics():
    """Calculate HDBSCAN validity metrics for the best trial"""
    import hdbscan
    
    # Get all cluster assignments and embeddings
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.cluster_id, p.embedding 
        FROM artifacts a
        JOIN papers p ON a.paper_id = p.id
        WHERE a.trial_id = %s AND a.cluster_id != -1
    ''', (best_trial,))
    results = cursor.fetchall()
    
    # Convert to numpy arrays
    labels = np.array([r[0] for r in results])
    embeddings = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in results])
    
    # Calculate validity metrics
    validity = hdbscan.validity.validity_index(
        X=embeddings,
        labels=labels,
        metric='euclidean',
        d=embeddings.shape[1],
        per_cluster_scores=True
    )
    
    # Get density separation between clusters
    unique_labels = np.unique(labels)
    density_separations = []
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            sep = hdbscan.validity.density_separation(
                X=embeddings,
                labels=labels,
                cluster_id1=unique_labels[i],
                cluster_id2=unique_labels[j],
                internal_nodes1=None,
                internal_nodes2=None,
                core_distances1=None,
                core_distances2=None
            )
            density_separations.append(sep)
    
    # Store per-cluster metrics in database
    with conn.cursor() as cursor:
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cluster_metrics (
                trial_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                validity_index REAL,
                density_separation REAL,
                PRIMARY KEY (trial_id, cluster_id)
            )
        ''')
        for cluster_id, validity_score in validity[1].items():
            cursor.execute('''
                INSERT INTO cluster_metrics 
                (trial_id, cluster_id, validity_index)
                VALUES (%s, %s, %s)
                ON CONFLICT (trial_id, cluster_id) DO UPDATE SET
                    validity_index = EXCLUDED.validity_index
            ''', (best_trial, cluster_id, validity_score))
        conn.commit()
    
    return {
        'overall_validity': validity[0],
        'mean_density_separation': np.mean(density_separations),
        'min_density_separation': np.min(density_separations)
    }

def calculate_and_store_validity_metrics():
    """Calculate and store validity metrics if missing"""
    if 'validity_index' not in best_trial_data['metrics']:
        # Calculate metrics
        validity_metrics = calculate_validity_metrics()
        
        # Update JSON data
        best_trial_data['metrics'].update({
            'validity_index': validity_metrics['overall_validity'],
            'mean_density_sep': validity_metrics['mean_density_separation'],
            'min_density_sep': validity_metrics['min_density_separation']
        })
        
        # Save to drive
        drive_path = "/content/drive/MyDrive/ai-safety-papers/best_trial.json"
        with open(drive_path, 'w') as f:
            json.dump(best_trial_data, f)
        
        # Print results
        print(f"Validity Index: {validity_metrics['overall_validity']:.3f}")
        print(f"Mean Density Separation: {validity_metrics['mean_density_separation']:.3f}")
        print(f"Min Density Separation: {validity_metrics['min_density_separation']:.3f}")
    else:
        print("Validity metrics already calculated")

def perform_2d_umap():
    """Generate and store 2D UMAP embeddings"""
    with conn.cursor() as cursor:
        cursor.execute('''
            ALTER TABLE artifacts 
            ADD COLUMN IF NOT EXISTS viz_embedding BYTEA
        ''')
        conn.commit()

        # Check if embeddings already exist
        cursor.execute('''
            SELECT COUNT(*) 
            FROM artifacts 
            WHERE trial_id = %s 
              AND viz_embedding IS NOT NULL
        ''', (best_trial,))
        if cursor.fetchone()[0] > 0:
            print("2D UMAP embeddings already exist")
            return
        
        # Load embeddings from papers table
        cursor.execute('''
            SELECT p.id, p.embedding 
            FROM papers p
            JOIN artifacts a ON p.id = a.paper_id
            WHERE a.trial_id = %s
        ''', (best_trial,))
        results = cursor.fetchall()
        
        # Convert to numpy array instead of cupy
        embeddings = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in results])
        
        # Use CPU-based UMAP implementation with trial parameters
        reducer = UMAP(
            n_components=2,  # Fixed for visualization
            n_neighbors=best_trial_data['params']['n_neighbors'],
        )
        viz_embeddings = reducer.fit_transform(embeddings)
        
        # Store results
        for i, (paper_id, _) in enumerate(results):
            cursor.execute('''
                UPDATE artifacts
                SET viz_embedding = %s
                WHERE trial_id = %s AND paper_id = %s
            ''', (viz_embeddings[i].tobytes(), best_trial, paper_id))
        
        conn.commit()
        print("2D UMAP embeddings generated and stored")
        
        # Create compressed backup
        backup_database()

def backup_database():
    """Create compressed database backup"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"
    print(f"Creating compressed backup at {backup_path}")
    !pg_dump -U postgres -F c -f "{backup_path}" papers # pyright: ignore
    print("Backup completed successfully")

# %% [markdown]
# ## Visualization Functions

# %%
def plot_clusters(figsize=(15, 15)):
    """Plot clusters with probability-based transparency"""

    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.paper_id, a.cluster_id, a.cluster_prob, a.viz_embedding
        FROM artifacts a
        WHERE a.trial_id = %s
    ''', (best_trial,))
    results = cursor.fetchall()
    
    # Convert embeddings directly to numpy
    coords = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in results])
    labels = np.array([r[1] for r in results])
    probabilities = np.array([r[2] for r in results])
    
    plt.figure(figsize=figsize)
    
    # Set up colors for clusters
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    
    # Plot noise points first
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(
            coords[noise_mask, 0],
            coords[noise_mask, 1],
            c='lightgray',
            marker='.',
            alpha=0.1,
            label='Noise'
        )
    
    # Plot clusters
    for i, label in enumerate(unique_labels):
        if label == -1:
            continue
            
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[colors[i]],
            marker='.',
            alpha=probabilities[mask],
            label=f'Cluster {label}'
        )
    
    plt.title('UMAP Visualization of Paper Clusters')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_cluster_metrics():
    """Plot metrics for the best clustering run"""
    # Get metrics from loaded JSON data
    metrics = best_trial_data['metrics']
    params = best_trial_data['params']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle(f'Clustering Trial {best_trial} Analysis')

    with conn.cursor() as cursor:
        # Cluster sizes plot
        cursor.execute('''
            SELECT cluster_id, COUNT(*) as size
            FROM artifacts
            WHERE trial_id = %s AND cluster_id >= 0
            GROUP BY cluster_id
        ''', (best_trial,))
        sizes = [row[1] for row in cursor.fetchall()]
        
    ax = axes[0, 0]
    ax.bar(range(len(sizes)), sizes)
    ax.set_title('Cluster Sizes')

    # Probability distribution plot
    with conn.cursor() as cursor:
        cursor.execute('''
            SELECT cluster_prob FROM artifacts
            WHERE trial_id = %s AND cluster_id >= 0
        ''', (best_trial,))
        probs = [row[0] for row in cursor.fetchall()]
    
    ax = axes[0, 1]
    sns.histplot(probs, bins=50, ax=ax)
    ax.set_title('Probability Distribution')

    # Metrics summary
    ax = axes[1, 0]
    metric_keys = ['noise_ratio', 'n_clusters', 'mean_persistence']
    metric_values = [metrics[k] for k in metric_keys]
    ax.bar([k.replace('_', ' ').title() for k in metric_keys], metric_values)
    ax.set_title('Quality Metrics')

    # Parameters summary
    ax = axes[1, 1]
    ax.axis('off')
    params_text = '\n'.join(f"{k}: {v}" for k,v in params.items())
    ax.text(0.1, 0.5, params_text, fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()

def plot_cluster_dendrogram(max_clusters=30):
    """Visualize cluster hierarchy using dendrogram"""
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT parent_cluster_id, child_cluster_id, lambda_val, child_size
        FROM cluster_trees
        WHERE trial_id = %s
        ORDER BY lambda_val DESC
    ''', (best_trial,))
    
    tree_data = cursor.fetchall()
    
    if not tree_data:
        print("No hierarchy data found for this run")
        return
    
    # Create distance matrix from tree
    all_nodes = set()
    for row in tree_data:
        all_nodes.add(row[0])
        all_nodes.add(row[1])
    
    n_points = len(all_nodes)
    distances = np.zeros((n_points, n_points))
    
    # Fill distance matrix based on lambda values
    for parent, child, lambda_val, _ in tree_data:
        distances[parent, child] = lambda_val
        distances[child, parent] = lambda_val
    
    # Convert to condensed form
    condensed_dist = squareform(distances)
    
    # Plot dendrogram
    plt.figure(figsize=(15, 10))
    dendrogram(
        condensed_dist,
        p=min(max_clusters, n_points),
        truncate_mode='lastp',
        show_leaf_counts=True
    )
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster')
    plt.ylabel('Distance (λ)')
    plt.show()

def plot_cluster_network(min_cluster_size=5):
    """Visualize cluster relationships as a network"""
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT parent_cluster_id, child_cluster_id, lambda_val, child_size
        FROM cluster_trees
        WHERE trial_id = %s
        ORDER BY lambda_val DESC
    ''', (best_trial,))
    tree_data = cursor.fetchall()
    
    if not tree_data:
        print("No hierarchy data found for this run")
        return
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Get lambda range for normalization
    lambda_vals = [row[2] for row in tree_data]
    lambda_range = max(lambda_vals) - min(lambda_vals)
    
    # Track clusters and their papers
    cluster_papers = {}
    
    # Add nodes and edges
    for parent, child, lambda_val, child_size in tree_data:
        if child_size > 1:  # Skip individual points
            # Normalize lambda value to [0,1] for visualization
            lambda_norm = (lambda_val - min(lambda_vals)) / lambda_range
            
            # Add nodes if they don't exist
            if parent not in G:
                G.add_node(parent, level=lambda_norm)
            if child not in G:
                G.add_node(child, level=lambda_norm)
            
            # Add edge
            G.add_edge(parent, child, weight=child_size)
            
            # Get papers for this cluster
            papers = get_papers_in_cluster(child)
            if papers:
                cluster_papers[child] = [
                    {'title': p[1], 'abstract': p[2]} 
                    for p in papers
                ]
    
    # Calculate layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create figure
    plt.figure(figsize=(20, 10))
    
    # Draw edges with varying thickness based on cluster size
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights)
    edge_widths = [2 * w / max_weight for w in edge_weights]
    
    # Draw the graph
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5)
    
    # Draw nodes with size based on number of papers
    node_sizes = []
    node_colors = []
    labels = {}
    
    for node in G.nodes():
        if node in cluster_papers:
            size = len(cluster_papers[node])
            node_sizes.append(1000 * size / len(cluster_papers))
            node_colors.append('lightblue')
            
            # Extract keywords for cluster label
            if size > min_cluster_size:
                keywords = extract_cluster_keywords(cluster_papers[node], n_keywords=3)
                labels[node] = '\n'.join([k[0] for k in keywords])
        else:
            node_sizes.append(100)
            node_colors.append('gray')
            labels[node] = ''
    
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes,
                          node_color=node_colors,
                          alpha=0.6)
    
    # Add labels with smaller font
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title('Cluster Hierarchy Network')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G, cluster_papers

def plot_hierarchical_clusters(figsize=(20, 10)):
    """Plot clusters at different hierarchical levels with interactive controls"""
    # Get all available lambda levels
    lambda_levels = get_cluster_hierarchy_levels()
    if not lambda_levels:
        print("No hierarchy levels found for this run")
        return
    
    # Get the 2D embeddings and original cluster assignments
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.paper_id, a.cluster_id, a.cluster_prob, a.viz_embedding
        FROM artifacts a
        WHERE a.trial_id = %s
    ''', (best_trial,))
    results = cursor.fetchall()
    
    # Convert embeddings to numpy array
    coords = np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in results])
    original_labels = np.array([r[1] for r in results])
    probabilities = np.array([r[2] for r in results])
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(2, 2, height_ratios=[1, 0.1])
    
    # Main scatter plot
    ax_scatter = fig.add_subplot(gs[0, :])
    
    # Slider axes
    ax_slider = fig.add_subplot(gs[1, :])
    
    # Initialize scatter plot
    scatter = None
    
    def update_plot(lambda_val):
        nonlocal scatter
        
        # Clear previous scatter plot
        if scatter is not None:
            scatter.remove()
        ax_scatter.clear()
        
        # Get clusters at this level
        clusters = get_clusters_at_lambda(lambda_val)
        cluster_ids = [c[0] for c in clusters]
        
        # Set up colors for clusters
        unique_labels = np.unique(cluster_ids)
        n_clusters = len(unique_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
        
        # Plot noise points first
        noise_mask = ~np.isin(original_labels, cluster_ids)
        if np.any(noise_mask):
            ax_scatter.scatter(
                coords[noise_mask, 0],
                coords[noise_mask, 1],
                c='lightgray',
                marker='.',
                alpha=0.1,
                label='Noise'
            )
        
        # Plot clusters
        for i, label in enumerate(unique_labels):
            mask = original_labels == label
            scatter = ax_scatter.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[colors[i]],
                marker='.',
                alpha=probabilities[mask],
                label=f'Cluster {label}'
            )
        
        # Get cluster info
        cluster_info = []
        for cluster_id in unique_labels:
            analysis = analyze_cluster_at_level(cluster_id, lambda_val)
            if analysis['keywords']:
                keywords = ", ".join(k[0] for k in analysis['keywords'][:3])
                cluster_info.append(f"Cluster {cluster_id}: {keywords}")
        
        # Update title with cluster information
        title = f'Clusters at λ = {lambda_val:.3f}\n'
        title += "\n".join(cluster_info)
        ax_scatter.set_title(title)
        
        # Add legend
        ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
    
    # Create slider
    lambda_min, lambda_max = min(lambda_levels), max(lambda_levels)
    slider = Slider(
        ax=ax_slider,
        label='Lambda Level',
        valmin=lambda_min,
        valmax=lambda_max,
        valinit=lambda_max,
        valstep=sorted(lambda_levels)
    )
    
    # Update function for slider
    def update(val):
        update_plot(slider.val)
    
    slider.on_changed(update)
    
    # Initial plot
    update_plot(lambda_max)
    
    plt.show()
    return fig

def print_cluster_hierarchy(max_depth=None):
    """Print the cluster hierarchy in a tree-like format"""
    cursor = conn.cursor()
    
    def print_cluster(cluster_id, depth=0, min_lambda=None):
        if max_depth is not None and depth >= max_depth:
            return
            
        # Get cluster info
        analysis = analyze_cluster_at_level(cluster_id, min_lambda or 0)
        
        # Print cluster details with indentation
        indent = "  " * depth
        print(f"{indent}Cluster {cluster_id} ({analysis['size']} papers)")
        
        if analysis['keywords']:
            print(f"{indent}Keywords:", ", ".join(k[0] for k in analysis['keywords'][:5]))
        
        # Get and sort children
        cursor.execute('''
            SELECT child_cluster_id, lambda_val
            FROM cluster_trees
            WHERE trial_id = %s AND parent_cluster_id = %s
            ORDER BY child_size DESC
        ''', (best_trial, cluster_id))
        
        children = cursor.fetchall()
        
        # Recursively print children
        for child_id, lambda_val in children:
            print_cluster(child_id, depth + 1, lambda_val)
    
    # Get root clusters
    cursor.execute('''
        SELECT DISTINCT parent_cluster_id
        FROM cluster_trees
        WHERE trial_id = %s
          AND parent_cluster_id NOT IN (
              SELECT child_cluster_id 
              FROM cluster_trees 
              WHERE trial_id = %s
          )
    ''', (best_trial, best_trial))
    
    root_clusters = [row[0] for row in cursor.fetchall()]
    
    # Print hierarchy starting from each root
    for root_id in root_clusters:
        print_cluster(root_id)

# %% [markdown]
## Example Usage

# %%
# Ensure 2D embeddings exist
perform_2d_umap()

# Calculate and store validity metrics
calculate_and_store_validity_metrics()

# Basic cluster visualization
plot_clusters()

# Cluster metrics and statistics
plot_cluster_metrics()

# Hierarchical structure
plot_cluster_dendrogram()
plot_cluster_network()
plot_hierarchical_clusters()
print_cluster_hierarchy(max_depth=2)
