import pandas as pd
import sqlite3
import os

def migrate_csv_to_sqlite():
    print("Initializing SQL Relational Database Layer...")
    
    # Check paths
    features_path = "data/processed_features.csv"
    if not os.path.exists(features_path):
        raise FileNotFoundError("Run feature pipeline script first to generate master data components.")
        
    # Read flat dataset
    master_df = pd.read_csv(features_path)
    
    # Establish connection to SQLite database file
    db_path = "data/bancassurance.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Clear preexisting data tables if running multiple times
    cursor.execute("DROP TABLE IF EXISTS customer_profiles")
    
    # Write dataframe schema straight to relational SQL
    print("Migrating flat features dataset to SQLite schema...")
    master_df.to_csv("data/temp_db.csv", index=False) # Clean temporary index dump
    master_df.to_sql("customer_profiles", conn, if_exists="replace", index=False)
    
    # Add an Explicit Primary Key Index to ensure production grade database design
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_cust_id ON customer_profiles (customer_id);")
    
    conn.commit()
    
    # Verify migration by querying counts
    cursor.execute("SELECT COUNT(*) FROM customer_profiles")
    total_records = cursor.fetchone()[0]
    
    conn.close()
    print(f"Migration Complete! Created database with {total_records} indexed relational customer entries.")

if __name__ == "__main__":
    migrate_csv_to_sqlite()