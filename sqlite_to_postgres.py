# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Database Conversion: SQLite to PostgreSQL
# This notebook converts the SQLite database to PostgreSQL format while preserving:
# - Schema structure
# - Data integrity
# - Relationships
# - Indexes

# %% [markdown]
# ## 1. Environment Setup

# %%
# Install required packages if running in Colab
import os
if 'COLAB_GPU' in os.environ:
    # Install PostgreSQL in Colab environment
    !sudo apt-get -qq update && sudo apt-get -qq install postgresql postgresql-contrib # pyright: ignore
    !sudo service postgresql start # pyright: ignore
    %pip install psycopg2-binary tqdm # pyright: ignore

# Mount Google Drive
from google.colab import drive # pyright: ignore [reportMissingImports]
drive.mount('/content/drive')

# %% [markdown]
# ## 2. Database Connections

# %%
# @title Database Credentials
postgres_host = "localhost" # @param {type:"string"}
postgres_db = "arxiv_papers" # @param {type:"string"}
postgres_user = "postgres" # @param {type:"string"}
postgres_password = "" # @param {type:"string"}

import sqlite3
import psycopg2
from tqdm import tqdm
import numpy as np

# Path to SQLite database
sqlite_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"

# Connect to SQLite
sqlite_conn = sqlite3.connect(sqlite_path)
sqlite_conn.row_factory = sqlite3.Row

# Connect to PostgreSQL
postgres_conn = psycopg2.connect(
    host=postgres_host,
    database=postgres_db,
    user=postgres_user,
    password=postgres_password
)
postgres_conn.autocommit = False
pg_cursor = postgres_conn.cursor()

# %% [markdown]
# ## 3. Schema Conversion

# %%
def create_postgres_schema(pg_cursor):
    """Create PostgreSQL schema matching SQLite structure"""
    try:
        # Create tables with PostgreSQL data types
        pg_cursor.execute('''
            CREATE TABLE papers (
                id TEXT PRIMARY KEY,
                title TEXT,
                abstract TEXT,
                categories TEXT,
                msc_class TEXT,
                acm_class TEXT,
                doi TEXT,
                license TEXT,
                comments TEXT,
                created TIMESTAMP,
                updated TIMESTAMP,
                withdrawn BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                abstract_embedding BYTEA
            )
        ''')
        
        pg_cursor.execute('''
            CREATE TABLE paper_versions (
                paper_id TEXT,
                version INTEGER,
                source_type TEXT,
                size TEXT,
                date TIMESTAMP,
                PRIMARY KEY (paper_id, version),
                FOREIGN KEY (paper_id) REFERENCES papers(id)
            )
        ''')
        
        pg_cursor.execute('''
            CREATE TABLE authors (
                id SERIAL PRIMARY KEY,
                keyname TEXT NOT NULL,
                forenames TEXT,
                suffix TEXT,
                CONSTRAINT unique_author UNIQUE (keyname, forenames, suffix)
            )
        ''')
        
        pg_cursor.execute('''
            CREATE TABLE paper_authors (
                paper_id TEXT,
                author_id INTEGER,
                author_position INTEGER,
                PRIMARY KEY (paper_id, author_id),
                FOREIGN KEY (paper_id) REFERENCES papers(id),
                FOREIGN KEY (author_id) REFERENCES authors(id)
            )
        ''')
        
        # Create indexes
        pg_cursor.execute('CREATE INDEX idx_categories ON papers(categories)')
        pg_cursor.execute('CREATE INDEX idx_withdrawn ON papers(withdrawn)')
        pg_cursor.execute('CREATE INDEX idx_created ON papers(created)')
        pg_cursor.execute('CREATE INDEX idx_updated ON papers(updated)')
        
        print("PostgreSQL schema created successfully")
        
    except psycopg2.Error as e:
        print(f"Error creating schema: {e}")
        postgres_conn.rollback()
        raise

# Create PostgreSQL schema
create_postgres_schema(pg_cursor)
postgres_conn.commit()

# %% [markdown]
# ## 4. Data Migration

# %%
def migrate_table(sqlite_conn, pg_cursor, table_name, columns, batch_size=1000):
    """Migrate data from SQLite to PostgreSQL with batch processing"""
    # Get total count for progress bar
    sqlite_cur = sqlite_conn.cursor()
    sqlite_cur.execute(f'SELECT COUNT(*) FROM {table_name}')
    total_rows = sqlite_cur.fetchone()[0]
    
    # Get data in batches
    offset = 0
    with tqdm(total=total_rows, desc=f"Migrating {table_name}", unit="rows") as pbar:
        while True:
            sqlite_cur.execute(f'SELECT * FROM {table_name} LIMIT ? OFFSET ?', (batch_size, offset))
            batch = sqlite_cur.fetchall()
            if not batch:
                break
                
            # Convert SQLite rows to PostgreSQL compatible format
            rows = []
            for row in batch:
                row_data = []
                for col in columns:
                    value = row[col]
                    # Convert SQLite boolean (0/1) to Python bool
                    if col == 'withdrawn' and value is not None:
                        value = bool(value)
                    # Handle numpy arrays for embeddings
                    if col == 'abstract_embedding' and value is not None:
                        value = psycopg2.Binary(value)
                    row_data.append(value)
                rows.append(tuple(row_data))
            
            # Generate INSERT query
            placeholders = ','.join(['%s'] * len(columns))
            columns_str = ','.join([f'"{col}"' for col in columns])
            query = f'INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})'
            
            try:
                pg_cursor.executemany(query, rows)
                postgres_conn.commit()
            except psycopg2.Error as e:
                print(f"Error inserting batch: {e}")
                postgres_conn.rollback()
                raise
                
            offset += len(batch)
            pbar.update(len(batch))

# Migration order respecting foreign key constraints
tables = [
    ('papers', ['id', 'title', 'abstract', 'categories', 'msc_class', 'acm_class',
                'doi', 'license', 'comments', 'created', 'updated', 'withdrawn',
                'created_at', 'abstract_embedding']),
    ('authors', ['id', 'keyname', 'forenames', 'suffix']),
    ('paper_versions', ['paper_id', 'version', 'source_type', 'size', 'date']),
    ('paper_authors', ['paper_id', 'author_id', 'author_position'])
]

# Disable foreign key constraints during migration
pg_cursor.execute('SET CONSTRAINTS ALL DEFERRED')

# Migrate tables
for table_name, columns in tables:
    migrate_table(sqlite_conn, pg_cursor, table_name, columns)

# Reset sequence for authors table
pg_cursor.execute("SELECT setval('authors_id_seq', (SELECT MAX(id) FROM authors))")
postgres_conn.commit()

# %% [markdown]
# ## 5. Data Validation

# %%
def validate_migration(sqlite_conn, postgres_conn):
    """Validate table counts and sample data between databases"""
    tables = ['papers', 'authors', 'paper_versions', 'paper_authors']
    
    for table in tables:
        # Get SQLite count
        sqlite_cur = sqlite_conn.cursor()
        sqlite_cur.execute(f'SELECT COUNT(*) FROM {table}')
        sqlite_count = sqlite_cur.fetchone()[0]
        
        # Get PostgreSQL count
        pg_cur = postgres_conn.cursor()
        pg_cur.execute(f'SELECT COUNT(*) FROM {table}')
        pg_count = pg_cur.fetchone()[0]
        
        print(f"{table}:")
        print(f"  SQLite: {sqlite_count}")
        print(f"  PostgreSQL: {pg_count}")
        print(f"  Match: {sqlite_count == pg_count}\n")
        
    # Check sample data
    print("\nSample Data Validation:")
    pg_cur.execute('''
        SELECT p.id, p.title, COUNT(a.id) as author_count
        FROM papers p
        JOIN paper_authors pa ON p.id = pa.paper_id
        JOIN authors a ON pa.author_id = a.id
        GROUP BY p.id
        LIMIT 5
    ''')
    print("PostgreSQL sample papers with author counts:")
    for row in pg_cur.fetchall():
        print(f"ID: {row[0]}, Authors: {row[2]}")
        
    sqlite_cur.execute('''
        SELECT p.id, p.title, COUNT(a.id) as author_count
        FROM papers p
        JOIN paper_authors pa ON p.id = pa.paper_id
        JOIN authors a ON pa.author_id = a.id
        GROUP BY p.id
        LIMIT 5
    ''')
    print("\nSQLite sample papers with author counts:")
    for row in sqlite_cur.fetchall():
        print(f"ID: {row[0]}, Authors: {row[2]}")

validate_migration(sqlite_conn, postgres_conn)

# %% [markdown]
# ## 6. Cleanup

# %%
# %% [markdown]
# ## 7. PostgreSQL Backup to Google Drive

# %%
def backup_postgres_db():
    """Backup PostgreSQL database to Google Drive"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/postgres_backup.sql"
    
    print(f"Creating PostgreSQL backup at {backup_path}")
    
    # Create backup using pg_dump
    !PGPASSWORD=$POSTGRES_PASSWORD pg_dump -h {postgres_host} -U {postgres_user} -d {postgres_db} -F c -f {backup_path} # pyright: ignore
    
    print("Backup completed successfully")

# Backup after successful migration
backup_postgres_db()

# Close connections
sqlite_conn.close()
postgres_conn.close()

print("Database conversion and backup completed successfully") 