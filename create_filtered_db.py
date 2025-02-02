# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Create Filtered AI Safety Database
# Creates a focused database containing:
# 1. Only CS.AI papers
# 2. Data from the best clustering trial
# 3. All related metadata and clustering artifacts

# %% [markdown]
# ## 1. Setup

# %%
# Mount Google Drive
from google.colab import drive # pyright: ignore
drive.mount('/content/drive')

# Install required packages
!sudo apt-get -qq update && sudo apt-get -qq install postgresql postgresql-contrib # pyright: ignore
!sudo service postgresql start # pyright: ignore
%pip install psycopg2-binary optuna # pyright: ignore

# %% [markdown]
# ## 2. Identify Best Trial

# %%
import psycopg2
import optuna
import json
import numpy as np

def get_optuna_storage():
    return "postgresql://postgres@/postgres"

# Add database loading from Drive backup
def load_database():
    """Load PostgreSQL backup using psql"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"
    print("Loading PostgreSQL backup...")
    !psql -U postgres -d postgres -f "{backup_path}" # pyright: ignore

# First load the database
load_database()

# Then connect to study
study = optuna.load_study(
    study_name="ai-papers-clustering",
    storage=get_optuna_storage()
)
best_trial = study.best_trial

# Save trial metadata
trial_info = {
    "trial_id": best_trial.number,
    "value": best_trial.value,
    "params": best_trial.params,
    "metrics": {k: float(v) if isinstance(v, np.generic) else v 
               for k, v in best_trial.user_attrs.items()}
}

with open("/content/drive/MyDrive/ai-safety-papers/best_trial.json", "w") as f:
    json.dump(trial_info, f, indent=2)

print(f"Best trial: {best_trial.number} with score {best_trial.value:.3f}")

# %% [markdown]
# ## 3. Create Filtered Database

# %%
# Database configuration
SOURCE_DB = "postgres"
DEST_DB = "papers"
BEST_TRIAL = best_trial.number

def execute_psql(command):
    """Execute PostgreSQL command using psql auth"""
    !psql -U postgres -d postgres -c "{command}" # pyright: ignore

# Create new database
execute_psql(f"DROP DATABASE IF EXISTS {DEST_DB}")
execute_psql(f"CREATE DATABASE {DEST_DB}")

# %% [markdown]
# ### 3.1 Export Schema

# %%
# Export schema only
!pg_dump -U postgres -d {SOURCE_DB} --schema-only -f /tmp/schema.sql # pyright: ignore
!psql -U postgres -d {DEST_DB} -f /tmp/schema.sql # pyright: ignore

# %% [markdown]
# ### 3.2 Export Filtered Data

# %%
# Export papers with embeddings (which implies cs.AI from embeddings.py)
!psql -U postgres -d {SOURCE_DB} -c "\copy (SELECT * FROM papers WHERE embedding IS NOT NULL AND withdrawn = FALSE) TO '/tmp/papers.sql'" # pyright: ignore
!psql -U postgres -d {DEST_DB} -c "\copy papers FROM '/tmp/papers.sql'" # pyright: ignore

# Export related authors (now based on embedding presence)
!psql -U postgres -d {SOURCE_DB} -c "\copy (SELECT * FROM authors WHERE id IN (SELECT author_id FROM paper_authors WHERE paper_id IN (SELECT id FROM papers WHERE embedding IS NOT NULL AND withdrawn = FALSE))) TO '/tmp/authors.sql'" # pyright: ignore
!psql -U postgres -d {DEST_DB} -c "\copy authors FROM '/tmp/authors.sql'" # pyright: ignore

# Export paper-author relationships
!psql -U postgres -d {SOURCE_DB} -c "\copy (SELECT * FROM paper_authors WHERE paper_id IN (SELECT id FROM papers WHERE embedding IS NOT NULL AND withdrawn = FALSE)) TO '/tmp/paper_authors.sql'" # pyright: ignore
!psql -U postgres -d {DEST_DB} -c "\copy paper_authors FROM '/tmp/paper_authors.sql'" # pyright: ignore

# Export paper versions
!psql -U postgres -d {SOURCE_DB} -c "\copy (SELECT * FROM paper_versions WHERE paper_id IN (SELECT id FROM papers WHERE embedding IS NOT NULL AND withdrawn = FALSE)) TO '/tmp/versions.sql'" # pyright: ignore
!psql -U postgres -d {DEST_DB} -c "\copy paper_versions FROM '/tmp/versions.sql'" # pyright: ignore

# Export best trial artifacts
!psql -U postgres -d {SOURCE_DB} -c "\copy (SELECT * FROM artifacts WHERE trial_id = {BEST_TRIAL}) TO '/tmp/artifacts.sql'" # pyright: ignore
!psql -U postgres -d {DEST_DB} -c "\copy artifacts FROM '/tmp/artifacts.sql'" # pyright: ignore

# Export cluster tree for best trial
!psql -U postgres -d {SOURCE_DB} -c "\copy (SELECT * FROM cluster_trees WHERE trial_id = {BEST_TRIAL}) TO '/tmp/cluster_trees.sql'" # pyright: ignore
!psql -U postgres -d {DEST_DB} -c "\copy cluster_trees FROM '/tmp/cluster_trees.sql'" # pyright: ignore

# %% [markdown]
# ## 4. Verify Database

# %%
def verify_database():
    conn = psycopg2.connect(
        host='',
        database=DEST_DB,
        user="postgres"
    )
    
    with conn.cursor() as c:
        # Core paper metrics
        c.execute("SELECT COUNT(*) FROM papers")
        total_papers = c.fetchone()[0]
        print(f"Total papers: {total_papers}")
        
        # Embedding coverage (from embeddings.py)
        c.execute("SELECT COUNT(*) FROM papers WHERE embedding IS NOT NULL")
        embedded_papers = c.fetchone()[0]
        print(f"\nEmbedding Coverage: {embedded_papers}/{total_papers} ({embedded_papers/total_papers:.1%})")
        
        # Citation tracking (from citations.py)
        c.execute("SELECT COUNT(*) FROM papers WHERE citation_count IS NOT NULL")
        cited_papers = c.fetchone()[0]
        print(f"Citation Data: {cited_papers}/{total_papers} ({cited_papers/total_papers:.1%})")

        # Version history (from harvesting.py)
        c.execute("SELECT COUNT(*) FROM paper_versions")
        versions = c.fetchone()[0]
        print(f"\nVersion History: {versions} total versions")
        c.execute("""
            SELECT AVG(version_count) 
            FROM (
                SELECT paper_id, COUNT(*) as version_count 
                FROM paper_versions 
                GROUP BY paper_id
            ) AS versions_subquery
        """)
        avg_versions = c.fetchone()[0]
        print(f"Average versions per paper: {avg_versions:.1f}")

        # Clustering artifacts (from clustering.py)
        c.execute(f"SELECT COUNT(*) FROM artifacts WHERE trial_id = {BEST_TRIAL}")
        artifacts = c.fetchone()[0]
        print(f"\nClustering Artifacts: {artifacts} (Trial {BEST_TRIAL})")
        
        # Cluster hierarchy (from clustering.py)
        c.execute(f"SELECT COUNT(*) FROM cluster_trees WHERE trial_id = {BEST_TRIAL}")
        cluster_edges = c.fetchone()[0]
        print(f"Cluster Hierarchy Edges: {cluster_edges}")

        # Author relationships (from harvesting.py)
        c.execute("""
            SELECT COUNT(DISTINCT a.id) 
            FROM authors a
            JOIN paper_authors pa ON a.id = pa.author_id
        """)
        active_authors = c.fetchone()[0]
        print(f"\nActive Authors: {active_authors}")
        
        c.execute("""
            SELECT COUNT(DISTINCT paper_id) 
            FROM paper_authors 
            WHERE paper_id IN (SELECT id FROM papers)
        """)
        papers_with_authors = c.fetchone()[0]
        print(f"Papers with author data: {papers_with_authors}/{total_papers} ({papers_with_authors/total_papers:.1%})")

        # Cluster metrics (from clustering optimization)
        c.execute(f"""
            SELECT COUNT(DISTINCT cluster_id), AVG(cluster_prob), STDDEV(cluster_prob)
            FROM artifacts 
            WHERE trial_id = {BEST_TRIAL} 
              AND cluster_id IS NOT NULL
        """)
        clusters, avg_prob, std_prob = c.fetchone()
        print(f"\nCluster Metrics (Trial {BEST_TRIAL}):")
        print(f"- Total clusters: {clusters}")
        print(f"- Average cluster probability: {avg_prob:.2f} ± {std_prob:.2f}")

        # Citation distribution analysis
        c.execute("""
            SELECT 
                COUNT(*) FILTER (WHERE citation_count > 0) AS non_zero,
                COUNT(*) FILTER (WHERE citation_count = 0) AS zero,
                COUNT(*) FILTER (WHERE citation_count IS NULL) AS nulls
            FROM papers
        """)
        non_zero, zero, nulls = c.fetchone()

        print(f"\nCitation Distribution:")
        print(f"- Papers with citations: {non_zero} ({non_zero/total_papers:.1%})")
        print(f"- Zero-citation papers: {zero} ({zero/total_papers:.1%})")
        print(f"- Unknown citation status: {nulls} ({nulls/total_papers:.1%})")

verify_database()

# %% [markdown]
# ## 5. Create Final Backup

# %%
!pg_dump -U postgres -d {DEST_DB} -F c -f /content/drive/MyDrive/ai-safety-papers/papers.sql # pyright: ignore
print("Filtered database backup created successfully") 
