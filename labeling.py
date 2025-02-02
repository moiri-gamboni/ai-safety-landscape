# %% [markdown]
# # AI Safety Paper Labeling Pipeline
# Stage 1: Individual Paper Processing

# %% [markdown]
# ## 1. Setup

# %%
# Mount Google Drive
from google.colab import drive # pyright: ignore [reportMissingImports]
drive.mount('/content/drive')

# Install required packages if running in Colab
import os
if 'COLAB_GPU' in os.environ:
    !sudo apt-get -qq update && sudo apt-get -qq install postgresql postgresql-contrib # pyright: ignore
    !sudo service postgresql start # pyright: ignore
    !sudo sed -i 's/local\s*all\s*postgres\s*peer/local all postgres trust/' /etc/postgresql/14/main/pg_hba.conf # pyright: ignore
    !sudo service postgresql restart # pyright: ignore
    
    %pip install psycopg2-binary tqdm tenacity google-genai # pyright: ignore

# %% [markdown]
# ## 2. Load Database

# %%
import psycopg2

VALID_CATEGORIES = [
    'cs.AI', 'cs.LG', 'cs.GT', 'cs.MA',
    'cs.LO', 'cs.CY', 'cs.CR', 'cs.SE', 'cs.NE'
]

def get_db_connection():
    """Create PostgreSQL connection"""
    return psycopg2.connect(
        host='',
        database="papers",
        user="postgres"
    )

def get_valid_categories():
    """Return as PostgreSQL array literal"""
    return "'{" + ",".join(VALID_CATEGORIES) + "}'"

def cleanup_database():
    """Permanently remove non-target papers and analysis tables"""
    with conn.cursor() as cursor:
        cursor.execute('SET maintenance_work_mem TO \'5GB\';')
    conn.commit()

    # Remove analysis tables
    with conn.cursor() as cursor:
        print("Removing old tables...")
        cursor.execute('DROP TABLE IF EXISTS alembic_version, artifacts, cluster_trees, studies, study_directions, study_system_attributes, study_user_attributes, trial_heartbeats, trial_intermediate_values, trial_params, trial_system_attributes, trial_user_attributes, trial_values, trials, version_info CASCADE')
        cursor.execute('DROP INDEX IF EXISTS idx_categories, idx_embedding_not_null, idx_updated, ix_studies_study_name, ix_trials_study_id')
    conn.commit()

    # Schema modifications
    with conn.cursor() as cursor:        
        print("Optimizing category filtering...")
        cursor.execute(f'''
            ALTER TABLE papers 
            ADD COLUMN IF NOT EXISTS arxiv_categories TEXT[]
            GENERATED ALWAYS AS (
                string_to_array(categories, ' ')
            ) STORED
        ''')
    conn.commit()

    with conn.cursor() as cursor:        
        print("Creating GIN index...")
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_paper_categories_arr 
            ON papers USING GIN(arxiv_categories)
        ''')
    conn.commit()

    with conn.cursor() as cursor:        
        print("Creating title index...")
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_title 
            ON papers USING GIN(to_tsvector('english', title))
        ''')
    conn.commit()

    # Delete related records
    with conn.cursor() as cursor:
        print("Deleting paper_versions records...")
        cursor.execute(f'''
            DELETE FROM paper_versions
            WHERE paper_id IN (
                SELECT id 
                FROM papers 
                WHERE (NOT arxiv_categories && {get_valid_categories()})
                   OR withdrawn = TRUE
            )
        ''')
    conn.commit()

    with conn.cursor() as cursor:
        print("Deleting paper_authors records...")
        cursor.execute(f'''
            DELETE FROM paper_authors
            WHERE paper_id IN (
                SELECT id FROM papers 
                WHERE (NOT arxiv_categories && {get_valid_categories()})
                   OR withdrawn = TRUE
            )
        ''')
    conn.commit()

    with conn.cursor() as cursor:
        print("Deleting non-target papers...")
        cursor.execute(f'''
            DELETE FROM papers
            WHERE (NOT arxiv_categories && {get_valid_categories()})
               OR withdrawn = TRUE
        ''')
    conn.commit()

    with conn.cursor() as cursor:
        print("Dropping original categories column...")
        cursor.execute('''
            ALTER TABLE papers 
            DROP COLUMN IF EXISTS categories
        ''')
    conn.commit()

    with conn.cursor() as cursor:
        print("Deleting orphaned authors...")
        cursor.execute('CREATE INDEX tmp_author_idx ON paper_authors (author_id)')
        cursor.execute('''
            DELETE FROM authors
            WHERE NOT EXISTS (
                SELECT 1 FROM paper_authors 
                WHERE author_id = authors.id
            )
        ''')
        cursor.execute('DROP INDEX tmp_author_idx')
        conn.commit()

    with conn.cursor() as cursor:
        print("Vacuuming database...")
        # Allow VACUUM outside transaction block
        conn.autocommit = True
        cursor.execute('VACUUM FULL ANALYZE')
        conn.autocommit = False
    
    with conn.cursor() as cursor:
        cursor.execute('RESET maintenance_work_mem;')
    conn.commit()
    
    print("Cleanup complete")

def load_database():
    """Load and filter PostgreSQL backup"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"
    print("Loading PostgreSQL backup...")
    !createdb -U postgres papers # pyright: ignore
    !pg_restore -U postgres --jobs=8 -d papers "{backup_path}" # pyright: ignore    

load_database()
conn = get_db_connection()
cleanup_database()

# %%
from google import genai
from tqdm.auto import tqdm
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
from psycopg2.extras import DictCursor
from typing import List, TypedDict
import time
from google.genai import types

# Configure Gemini API
if 'COLAB_GPU' in os.environ:
    # @title Gemini API Key
    gemini_api_key = "" # @param {type:"string"}
    client = genai.Client(api_key=gemini_api_key)
else:
    from dotenv import load_dotenv
    load_dotenv()
    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

MODEL_ID = "gemini-2.0-flash-exp"

# %%
def create_label_columns():
    """Create columns for labeling results matching db_schema.md"""
    with conn.cursor() as cursor:
        cursor.execute('''
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'papers' 
            AND column_name IN ('llm_category', 'safety_relevance', 'label_confidence')
        ''')
        existing_columns = {row[0] for row in cursor.fetchall()}
        
        if 'llm_category' not in existing_columns:
            cursor.execute('ALTER TABLE papers ADD COLUMN llm_category TEXT')
        if 'safety_relevance' not in existing_columns:
            cursor.execute('ALTER TABLE papers ADD COLUMN safety_relevance FLOAT')
        if 'label_confidence' not in existing_columns:
            cursor.execute('ALTER TABLE papers ADD COLUMN label_confidence FLOAT')
        conn.commit()

# Create columns before processing
create_label_columns()

# %%
def get_paper_batches(batch_size=400):
    """Generator yielding batches of papers needing labeling"""
    cursor = conn.cursor(cursor_factory=DictCursor)
    
    # Get total count of papers needing labels
    cursor.execute('''
        SELECT COUNT(*) 
        FROM papers 
        WHERE llm_category IS NULL
          AND title IS NOT NULL
          AND abstract IS NOT NULL
    ''')
    
    total_papers = cursor.fetchone()[0]
    
    # Simplified query without category filtering
    cursor.execute('''
        SELECT id, title, abstract 
        FROM papers
        WHERE llm_category IS NULL
          AND title IS NOT NULL
          AND abstract IS NOT NULL
        ORDER BY id
    ''')
    
    batch = []
    with tqdm(total=total_papers, desc="Processing papers", unit=" papers") as pbar:
        for row in cursor:
            batch.append(row)
            if len(batch) >= batch_size:
                pbar.update(len(batch))
                yield batch
                batch = []
        if batch:
            pbar.update(len(batch))
            yield batch

# %%
class PaperLabel(TypedDict):
    category: str
    relevance_score: float
    confidence: float

model = genai.GenerativeModel("models/gemini-2.0-flash-exp")
generation_config = genai.GenerationConfig(
    response_mime_type="application/json",
    response_schema=List[PaperLabel],
)

# %%
# Rate limiter class matching Gemini 1.5 Flash limits
class GeminiRateLimiter:
    def __init__(self):
        self.rpm_limit = 10  # Requests per minute
        self.tpm_limit = 4_000_000  # Tokens per minute
        self.rpd_limit = 1_500  # Requests per day
        self.requests = []
        self.tokens = []
        self.daily_count = 0
        
    def can_make_request(self, input_tokens, output_tokens):
        current_time = time.time()
        cutoff_time = current_time - 60
        
        # Check daily limit
        if self.daily_count >= self.rpd_limit:
            print("Daily request limit reached")
            return False
            
        # Check RPM
        recent_requests = [t for t in self.requests if t > cutoff_time]
        if len(recent_requests) >= self.rpm_limit:
            print(f"RPM limit reached ({len(recent_requests)}/{self.rpm_limit})")
            return False
            
        # Check TPM
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
        self.daily_count += 1

# Initialize rate limiter
rate_limiter = GeminiRateLimiter()

# %%
def test_token_count():
    """Test token counting with real data batch"""
    try:
        batch = next(get_paper_batches(batch_size=400))
    except StopIteration:
        print("No unprocessed papers available for testing")
        return

    # Use generate_labels to test full flow
    labels, input_toks, output_toks = generate_labels(batch)
    
    print(f"\nTest batch results:")
    print(f"Input tokens: {input_toks}")
    print(f"Output tokens: {output_toks}")
    print(f"Total tokens: {input_toks + output_toks}")
    
    # Check against limits
    if output_toks >= 8192:
        print("\n⚠️ WARNING: Exceeded output token limit! Response may be truncated")
    else:
        print(f"\n✅ Output tokens within limit (8192)")

    return input_toks, output_toks

# Run updated test
test_token_count()

# %%
class TokenTracker:
    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.batches_processed = 0
        
    def add_batch(self, input_tokens, output_tokens):
        self.total_input += input_tokens
        self.total_output += output_tokens
        self.batches_processed += 1
        
    def print_summary(self):
        print("\n=== Token Usage Summary ===")
        print(f"Total batches processed: {self.batches_processed}")
        print(f"Total input tokens: {self.total_input}")
        print(f"Total output tokens: {self.total_output}")
        print(f"Total tokens used: {self.total_input + self.total_output}")

token_tracker = TokenTracker()

# %%
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def generate_labels(batch):
    """Generate labels for a batch of papers using Gemini"""
    base_prompt = """You are an expert in AI safety and machine learning. Your task is to categorize academic papers and assess their relevance to AI safety.

For each paper, provide:
1. A specific technical category that precisely describes the primary research focus (e.g. "Adversarial Attack Detection" rather than "AI Safety", or "Reward Modeling for RLHF" rather than "Reinforcement Learning")
2. A relevance score (0-1) indicating how relevant it is to AI safety research
3. Your confidence (0-1) in this categorization

Guidelines:
- Use precise technical terminology
- Categories should be specific enough to differentiate between similar papers
- Consider both direct and indirect relevance to AI safety
- Be consistent in scoring across papers

Papers to analyze:
"""
    
    prompt = base_prompt
    for paper in batch:
        prompt += f"\n\nTitle: {paper['title']}\nAbstract: {paper['abstract']}"
    
    # Count tokens
    count_response = client.models.count_tokens(
        model=MODEL_ID,
        contents=prompt
    )
    input_tokens = count_response.total_tokens
    
    # Check rate limits (using actual output tokens)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        generation_config=generation_config
    )
    
    if not response.usage_metadata:
        raise ValueError("Missing usage metadata in response")
    
    output_tokens = response.usage_metadata.candidates_token_count
    
    # Update rate limiter and tracker
    rate_limiter.add_request(input_tokens, output_tokens)
    token_tracker.add_batch(input_tokens, output_tokens)
    
    # Add truncation check
    if output_tokens >= 8192:
        print(f"Warning: Output tokens at limit ({output_tokens}), response may be truncated")
    
    return json.loads(response.text), input_tokens, output_tokens

# %%
def process_batches():
    """Main processing loop with rate limiting"""
    for batch in get_paper_batches():
        try:
            labels, input_toks, output_toks = generate_labels(batch)
            
            # Print batch stats
            print(f"Processed batch of {len(batch)} papers")
            print(f"Tokens: {input_toks} in → {output_toks} out")
            
            # Update database
            with conn.cursor() as cursor:
                for paper, label in zip(batch, labels):
                    cursor.execute('''
                        UPDATE papers
                        SET llm_category = %s,
                            safety_relevance = %s,
                            label_confidence = %s
                        WHERE id = %s
                    ''', (
                        label.get('category'),
                        label.get('relevance_score'),
                        label.get('confidence'),
                        paper['id']
                    ))
                conn.commit()
                
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            conn.rollback()
    
    # Print final summary
    token_tracker.print_summary()

# %%
# Execute the pipeline
process_batches()

# %% [markdown]
# ## Data Validation

# %%
def validate_labels():
    """Validate labeling results"""
    cursor = conn.cursor(cursor_factory=DictCursor)
    
    print("\n=== Labeling Quality Checks ===")
    
    # Check coverage of labeling process
    cursor.execute('''
        SELECT 
            COUNT(*) AS total,
            SUM(CASE WHEN llm_category IS NOT NULL THEN 1 ELSE 0 END) AS labeled,
            SUM(CASE WHEN llm_category IS NULL THEN 1 ELSE 0 END) AS unlabeled
        FROM papers
    ''')
    
    stats = cursor.fetchone()
    print(f"Label coverage: {stats['labeled']}/{stats['total']} ({stats['labeled']/stats['total']*100:.1f}%)")

validate_labels()

# %% [markdown]
# ## Execution Flow

# %%
# Run entire pipeline
process_batches()
validate_labels()


# %% [markdown]
# ## 6. Save Results

# %%
# Save duplicates to Drive
def backup_database():
    """Backup entire (now filtered) database"""
    backup_path = "/content/drive/MyDrive/ai-safety-papers/papers.sql"
    !pg_dump -U postgres -F c -f "{backup_path}" papers # pyright: ignore
    print(f"Full filtered backup saved to {backup_path}")

# Call backup after processing
backup_database()

