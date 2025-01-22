# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # AI Safety Papers Visualization - Cleanup (should not be needed after fixes)

# %% [markdown]
# ## Load Existing Database

# %%
import sqlite3
import os

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Check if database exists in Drive
db_path = "/content/drive/MyDrive/ai-safety-papers/papers.db"
if os.path.exists(db_path):
    print(f"Found existing database at {db_path}")
    !cp "{db_path}" papers.db
    
    # Print existing data summary
    conn = sqlite3.connect('papers.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM papers')
    print(f"Database contains {c.fetchone()[0]} papers")
else:
    raise Exception("No existing database found in Drive.")

# %% [markdown]
# ## Clean Up Author Duplicates
# Run this cell to deduplicate authors and fix the database schema.

# %%
def cleanup_authors(conn):
    """Clean up duplicate authors and add proper constraints"""
    c = conn.cursor()
    print("Starting author cleanup...")
    
    try:
        # Start transaction
        c.execute('BEGIN TRANSACTION')
        
        # Create temporary table for canonical authors with pre-computed normalized values
        print("Creating canonical authors table...")
        c.execute('DROP TABLE IF EXISTS canonical_authors')
        c.execute('''
            CREATE TABLE canonical_authors AS
            SELECT 
                MIN(id) as canonical_id,
                keyname,
                forenames,
                CASE 
                    WHEN suffix IS NOT NULL AND UPPER(suffix) IN ('JR', 'JR.', 'JR ', 'JUNIOR') THEN 'Jr.'
                    WHEN suffix IS NOT NULL AND UPPER(suffix) IN ('I', 'II', 'III', 'IV', 'V') THEN UPPER(suffix)
                    ELSE suffix
                END as suffix,
                COALESCE(forenames, '') as forenames_clean,
                COALESCE(
                    CASE 
                        WHEN suffix IS NOT NULL AND UPPER(suffix) IN ('JR', 'JR.', 'JR ', 'JUNIOR') THEN 'Jr.'
                        WHEN suffix IS NOT NULL AND UPPER(suffix) IN ('I', 'II', 'III', 'IV', 'V') THEN UPPER(suffix)
                        ELSE suffix
                    END,
                    ''
                ) as suffix_clean
            FROM authors
            GROUP BY 
                keyname, 
                forenames,
                CASE 
                    WHEN suffix IS NOT NULL AND UPPER(suffix) IN ('JR', 'JR.', 'JR ', 'JUNIOR') THEN 'Jr.'
                    WHEN suffix IS NOT NULL AND UPPER(suffix) IN ('I', 'II', 'III', 'IV', 'V') THEN UPPER(suffix)
                    ELSE suffix
                END
        ''')
        
        # Create indices for faster joins using normalized columns
        print("Creating indices for mapping...")
        c.execute('CREATE INDEX idx_canonical_lookup ON canonical_authors(keyname, forenames_clean, suffix_clean)')
        c.execute('CREATE INDEX idx_authors_lookup ON authors(keyname)')
        
        # Create mapping table with proper schema
        print("Creating author ID mapping table...")
        c.execute('DROP TABLE IF EXISTS author_id_mapping')
        c.execute('''
            CREATE TABLE author_id_mapping (
                old_id INTEGER PRIMARY KEY,
                new_id INTEGER NOT NULL
            )
        ''')
        
        # Process authors in batches with pre-computed normalized values
        print("\nBuilding author ID mapping in batches...")
        batch_size = 100000
        c.execute('SELECT COUNT(*) FROM authors')
        total_authors = c.fetchone()[0]
        processed = 0
        
        while processed < total_authors:
            c.execute('''
                WITH normalized_batch AS (
                    SELECT 
                        id,
                        keyname,
                        COALESCE(forenames, '') as forenames_clean,
                        COALESCE(
                            CASE 
                                WHEN suffix IS NOT NULL AND UPPER(suffix) IN ('JR', 'JR.', 'JR ', 'JUNIOR') THEN 'Jr.'
                                WHEN suffix IS NOT NULL AND UPPER(suffix) IN ('I', 'II', 'III', 'IV', 'V') THEN UPPER(suffix)
                                ELSE suffix
                            END,
                            ''
                        ) as suffix_clean
                    FROM authors
                    WHERE rowid > ? AND rowid <= ?
                )
                INSERT INTO author_id_mapping (old_id, new_id)
                SELECT 
                    nb.id as old_id,
                    ca.canonical_id as new_id
                FROM normalized_batch nb
                JOIN canonical_authors ca ON 
                    nb.keyname = ca.keyname AND
                    nb.forenames_clean = ca.forenames_clean AND
                    nb.suffix_clean = ca.suffix_clean
            ''', (processed, processed + batch_size))
            
            processed += batch_size
            print(f"Progress: {min(processed, total_authors)}/{total_authors} authors mapped ({(min(processed, total_authors)/total_authors*100):.1f}%)")
        
        # Create index for faster lookups
        print("Creating index on mapping table...")
        c.execute('CREATE INDEX idx_old_id ON author_id_mapping(old_id)')
        
        # Print mapping stats and verify coverage
        print("\nChecking mapping coverage...")
        c.execute('SELECT COUNT(*) FROM author_id_mapping')
        total_mappings = c.fetchone()[0]
        print(f"Total authors: {total_authors}")
        print(f"Total mappings: {total_mappings}")
        
        if total_mappings != total_authors:
            c.execute('''
                SELECT a.id, a.keyname, a.forenames, a.suffix
                FROM authors a
                LEFT JOIN author_id_mapping m ON a.id = m.old_id
                WHERE m.new_id IS NULL
                LIMIT 5
            ''')
            print("\nSample unmapped authors:")
            for row in c.fetchall():
                print(f"ID {row[0]}: {row[1]}, {row[2]}, {row[3]}")
            raise Exception(f"Found {total_authors - total_mappings} unmapped authors")
        
        # Update paper_authors in batches
        print("\nUpdating paper-author links...")
        c.execute('SELECT COUNT(*) FROM paper_authors')
        total_links = c.fetchone()[0]
        processed = 0
        
        while processed < total_links:
            c.execute('''
                UPDATE paper_authors
                SET author_id = (
                    SELECT new_id
                    FROM author_id_mapping
                    WHERE old_id = paper_authors.author_id
                )
                WHERE rowid IN (
                    SELECT rowid FROM paper_authors
                    LIMIT ? OFFSET ?
                )
            ''', (batch_size, processed))
            
            processed += batch_size
            print(f"Progress: {min(processed, total_links)}/{total_links} links updated ({(min(processed, total_links)/total_links*100):.1f}%)")
        
        # Verify all paper_authors entries have valid author_ids
        print("\nVerifying paper-author links...")
        c.execute('''
            SELECT COUNT(*) FROM paper_authors pa
            LEFT JOIN author_id_mapping m ON pa.author_id = m.old_id
            WHERE m.new_id IS NULL
        ''')
        orphaned = c.fetchone()[0]
        if orphaned > 0:
            raise Exception(f"Found {orphaned} paper-author links that would be orphaned")
        
        # Replace authors table with canonical version
        print("Replacing authors table with deduplicated version...")
        c.execute('DROP TABLE IF EXISTS authors_backup')
        c.execute('ALTER TABLE authors RENAME TO authors_backup')
        c.execute('''
            CREATE TABLE authors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyname TEXT NOT NULL,
                forenames TEXT,
                suffix TEXT,
                UNIQUE(keyname, forenames, suffix)
            )
        ''')
        
        # Insert canonical authors
        c.execute('''
            INSERT INTO authors (id, keyname, forenames, suffix)
            SELECT canonical_id, keyname, forenames, suffix
            FROM canonical_authors
        ''')
        
        # Print results
        c.execute('SELECT COUNT(*) FROM authors_backup')
        authors_before = c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM authors')
        authors_after = c.fetchone()[0]
        print(f"\nCleanup complete:")
        print(f"Authors before: {authors_before}")
        print(f"Authors after: {authors_after}")
        print(f"Duplicates removed: {authors_before - authors_after}")
        
        # Cleanup temporary tables
        print("\nCleaning up temporary tables...")
        c.execute('DROP TABLE IF EXISTS canonical_authors')
        c.execute('DROP TABLE IF EXISTS author_id_mapping')
        c.execute('DROP TABLE IF EXISTS authors_backup')
        
        conn.commit()
        print("Changes committed successfully")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        print("Rolling back changes...")
        conn.rollback()
        
        # Cleanup temporary tables
        c.execute('DROP TABLE IF EXISTS canonical_authors')
        c.execute('DROP TABLE IF EXISTS author_id_mapping')
        
        # Restore original authors table if needed
        c.execute('SELECT COUNT(*) FROM authors')
        if c.fetchone()[0] == 0:
            print("Restoring authors table from backup...")
            c.execute('DROP TABLE authors')
            c.execute('ALTER TABLE authors_backup RENAME TO authors')
            conn.commit()
            print("Original authors table restored")
        raise

# Run cleanup if needed
cleanup_authors(conn)

# %% [markdown]
# ## Clean Up Author Commas
# Run this cell to fix author keynames that incorrectly include trailing commas.

# %%
def cleanup_author_commas(conn):
    """Clean up author keynames that incorrectly include trailing commas by merging with existing authors"""
    c = conn.cursor()
    print("Starting author comma cleanup...")
    
    try:
        # Start transaction
        c.execute('BEGIN TRANSACTION')
        
        # Find authors with trailing commas in keyname
        c.execute('''
            SELECT id, keyname, forenames, suffix
            FROM authors
            WHERE keyname LIKE '%,'
        ''')
        authors_to_fix = c.fetchall()
        
        if not authors_to_fix:
            print("No authors found with trailing commas in keyname")
            return
            
        print(f"\nFound {len(authors_to_fix)} authors with trailing commas in keyname")
        
        # Create temporary mapping table
        c.execute('DROP TABLE IF EXISTS author_comma_mapping')
        c.execute('''
            CREATE TABLE author_comma_mapping (
                old_id INTEGER PRIMARY KEY,
                new_id INTEGER NOT NULL
            )
        ''')
        
        # Process each author
        for author_id, keyname, forenames, suffix in authors_to_fix:
            fixed_keyname = keyname.rstrip(',')
            print(f"\nProcessing author {author_id}:")
            print(f"  Before: keyname='{keyname}', forenames='{forenames or ''}', suffix='{suffix or ''}'")
            print(f"  After:  keyname='{fixed_keyname}', forenames='{forenames or ''}', suffix='{suffix or ''}'")
            
            # Check if normalized version already exists
            c.execute('''
                SELECT id FROM authors 
                WHERE keyname = ? AND 
                      COALESCE(forenames, '') = COALESCE(?, '') AND
                      COALESCE(suffix, '') = COALESCE(?, '')
            ''', (fixed_keyname, forenames, suffix))
            
            existing = c.fetchone()
            if existing:
                existing_id = existing[0]
                print(f"  Found existing author with ID {existing_id}")
                
                # Map old ID to existing ID
                c.execute('INSERT INTO author_comma_mapping (old_id, new_id) VALUES (?, ?)',
                         (author_id, existing_id))
                
                # Update paper_authors to use existing ID
                c.execute('''
                    UPDATE OR IGNORE paper_authors 
                    SET author_id = ? 
                    WHERE author_id = ?
                ''', (existing_id, author_id))
                
                # Update author_affiliations to use existing ID
                c.execute('''
                    INSERT OR IGNORE INTO author_affiliations (author_id, affiliation)
                    SELECT ?, affiliation
                    FROM author_affiliations
                    WHERE author_id = ?
                ''', (existing_id, author_id))
                
                # Delete old author record and their affiliations
                c.execute('DELETE FROM author_affiliations WHERE author_id = ?', (author_id,))
                c.execute('DELETE FROM authors WHERE id = ?', (author_id,))
            else:
                # Just update the keyname if no existing author found
                c.execute('''
                    UPDATE authors
                    SET keyname = ?
                    WHERE id = ?
                ''', (fixed_keyname, author_id))
        
        # Cleanup
        c.execute('DROP TABLE IF EXISTS author_comma_mapping')
        
        conn.commit()
        print("\nChanges committed successfully")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        print("Rolling back changes...")
        conn.rollback()
        
        # Cleanup temporary table
        c.execute('DROP TABLE IF EXISTS author_comma_mapping')
        raise

# Run cleanup if needed
cleanup_author_commas(conn)

# %% [markdown]
# ## Save Database to Drive

# %%
!cp papers.db "/content/drive/MyDrive/ai-safety-papers/papers.db"
print("Database saved to Google Drive at: /ai-safety-papers/papers.db") 