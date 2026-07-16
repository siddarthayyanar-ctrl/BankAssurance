# Database Migration Script Documentation

## Overview

The file [src/db_migration.py](../src/db_migration.py) migrates preprocessed customer features from a flat CSV file into a production-grade SQLite relational database. It transforms `data/processed_features.csv` into an indexed `data/bancassurance.db` with optimized schema design for fast lookups and ACID compliance.

This script is the bridge between feature engineering and the API layer, ensuring that [app/main.py](../app/main.py) can query customer profiles efficiently at runtime.

---

## Purpose

The script performs three main tasks:

1. **Data Migration**: Loads CSV features into SQLite with proper schema inference
2. **Index Optimization**: Creates a unique index on `customer_id` for O(1) lookup performance
3. **Validation**: Verifies migration success by counting total records

---

## Functional Workflow

### 1. Path Validation

The script begins by checking that the source data file exists:

```python
features_path = "data/processed_features.csv"
if not os.path.exists(features_path):
    raise FileNotFoundError("Run feature pipeline script first...")
```

If `data/processed_features.csv` is missing, the script raises a `FileNotFoundError` with instructions to run the feature pipeline first.

### 2. CSV Loading

The flat feature dataset is loaded into memory:

```python
master_df = pd.read_csv(features_path)
```

This DataFrame contains all customer records with the following expected columns:
- Demographics: `age`, `annual_income`, `credit_score`, `tenure_months`
- Financial: `current_account_balance`, `total_salary_credited`, `total_debit_amount`
- Behavioral: `spend_Auto_Service`, `spend_Dining`, `spend_Fuel`, `spend_Groceries`, `spend_Medical`, `spend_Pharmacy`, `spend_Real_Estate`, `spend_Utilities`
- Transactional: `avg_transaction_size`, `transaction_count`
- Targets: `has_auto_ins`, `has_health_ins`, `has_life_ins`, `actual_12m_value`
- Identity: `customer_id`

### 3. Database Connection & Cleanup

The script establishes a SQLite connection and drops any preexisting tables:

```python
conn = sqlite3.connect(db_path)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS customer_profiles")
```

This ensures a clean state for repeated migrations (useful in development/testing workflows).

### 4. Schema Creation & Data Insert

Pandas automatically infers SQL types and creates the `customer_profiles` table:

```python
master_df.to_sql("customer_profiles", conn, if_exists="replace", index=False)
```

The `if_exists="replace"` parameter handles the table creation after the DROP operation.

### 5. Index Creation

A unique index is created on the `customer_id` column for O(1) lookups:

```python
cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_cust_id ON customer_profiles (customer_id);")
```

**Why unique?** Ensures `customer_id` is a valid primary key with no duplicates.

**Performance Impact**: Enables FastAPI `/api/v2/predict` endpoint to retrieve individual customer records instantly.

### 6. Commit & Verification

The transaction is committed and record count is validated:

```python
conn.commit()
cursor.execute("SELECT COUNT(*) FROM customer_profiles")
total_records = cursor.fetchone()[0]
conn.close()
print(f"Migration Complete! Created database with {total_records} indexed relational customer entries.")
```

---

## Input Files

- **`data/processed_features.csv`** — Output from feature engineering pipeline
  - Required columns: All columns listed in section 2 above
  - Row count: Arbitrary (tested with 1K-100K customer records)
  - Delimiter: Comma (`.csv`)

---

## Output Files

- **`data/bancassurance.db`** — SQLite database file
  - Table: `customer_profiles` (same schema as CSV)
  - Index: `idx_cust_id` on `customer_id` column
  - Size: Approximately 2-3x smaller than the CSV source

- **`data/temp_db.csv`** — Temporary debugging file
  - Created by `master_df.to_csv("data/temp_db.csv", index=False)`
  - Safe to delete after successful migration
  - Used for verification and troubleshooting

---

## How to Run

From the repository root:

```bash
python3 src/db_migration.py
```

**Expected Output**:
```
Initializing SQL Relational Database Layer...
Migrating flat features dataset to SQLite schema...
Migration Complete! Created database with 5000 indexed relational customer entries.
```

---

## Key Design Decisions

- **Unique Index on `customer_id`**: Guarantees primary key constraint and enables fast single-customer queries.
- **DROP TABLE IF EXISTS**: Allows safe re-runs without manual cleanup; important for dev/test cycles.
- **Temporary CSV dump**: `data/temp_db.csv` provides a verification artifact (can be removed).
- **No additional transformations**: Schema directly mirrors CSV structure; all feature engineering is pre-migration.
- **SQLite over PostgreSQL/MySQL**: Lightweight, file-based, no server required for local development/deployment.

---

## Dependencies

- `pandas` — CSV reading and DataFrame to SQL conversion
- `sqlite3` — Database connection and SQL operations
- `os` — File path validation

---

## Integration with API Layer

Once migration completes successfully, [app/main.py](../app/main.py) uses the database for:

1. **`GET /api/v2/meta`** — Aggregates across all `customer_profiles` rows
2. **`POST /api/v2/predict`** — Queries single customer by indexed `customer_id`

The index created by this script ensures that `SELECT * FROM customer_profiles WHERE customer_id = ?` runs in constant time.

---

## Troubleshooting

**FileNotFoundError: `data/processed_features.csv` not found**
- Run the feature engineering pipeline first (e.g., `python3 src/features.py`)
- Verify the file exists and is readable

**Permission Error on `data/bancassurance.db`**
- Check write permissions in the `data/` directory
- Ensure the database file is not locked by another process

**Duplicate `customer_id` values cause index creation to fail**
- Feature pipeline bug: customer IDs should be unique
- Verify the CSV has no duplicate rows
- Run deduplication before migration: `df.drop_duplicates(subset=['customer_id'], inplace=True)`

**API cannot connect to database**
- Ensure migration completed successfully (check terminal output)
- Verify `data/bancassurance.db` exists and is readable
- Check that [app/main.py](../app/main.py) uses correct path: `DB_PATH = "data/bancassurance.db"`

---

## Verification Steps

After migration, validate the database:

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('data/bancassurance.db')
cursor = conn.cursor()

# Check table exists
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='table' AND name='customer_profiles'\")
print(f'Table exists: {cursor.fetchone() is not None}')

# Check record count
cursor.execute('SELECT COUNT(*) FROM customer_profiles')
print(f'Total records: {cursor.fetchone()[0]}')

# Check index exists
cursor.execute(\"SELECT name FROM sqlite_master WHERE type='index' AND name='idx_cust_id'\")
print(f'Index created: {cursor.fetchone() is not None}')

conn.close()
"
```

Expected output:
```
Table exists: True
Total records: 5000
Index created: True
```
