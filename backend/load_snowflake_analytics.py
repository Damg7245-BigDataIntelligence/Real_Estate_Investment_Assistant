import os
import snowflake.connector
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def snowflake_pipeline():
    """
    Loads CSV file(s) from the S3 folder "Merged_Files" into Snowflake.
    This script:
      - Connects to Snowflake using credentials from the environment.
      - Ensures the warehouse, database, and schema (REAL_ESTATE_SCHEMA) exist.
      - Creates a file format and an external stage pointing to the S3 "Merged_Files" folder.
      - Lists files in the stage to ensure they exist.
      - Creates a target table with a fixed definition.
      - Uses a COPY command with a PATTERN to load all CSV files from that folder.
    """
    # Snowflake connection configuration
    conn_config_sf = {
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "role": os.getenv("SNOWFLAKE_ROLE")
    }
    
    # S3 details for merged file(s)
    s3_bucket = os.getenv("AWS_S3_BUCKET_NAME")
    s3_merged_folder = "Merged_Files"  # Folder in S3 where merged CSV file(s) reside
    s3_full_path = f"s3://{s3_bucket}/{s3_merged_folder}/"
    print(f"Using S3 folder: {s3_full_path}")
    
    try:
        with snowflake.connector.connect(**conn_config_sf) as conn:
            with conn.cursor() as cur:
                # Ensure warehouse, database, and schema exist; then use REAL_ESTATE_SCHEMA.
                cur.execute("""
                    CREATE WAREHOUSE IF NOT EXISTS BOSTON_REAL_ESTATE_WH
                    WAREHOUSE_SIZE = 'SMALL'
                    AUTO_SUSPEND = 60
                    AUTO_RESUME = TRUE;
                """)
                cur.execute("CREATE DATABASE IF NOT EXISTS SUFFOLK_REAL_ESTATE_DB;")
                cur.execute("USE DATABASE SUFFOLK_REAL_ESTATE_DB;")
                
                # Create the schema if it doesn't exist
                cur.execute("CREATE SCHEMA IF NOT EXISTS SUFFOLK_ANALYTICS_SCHEMA;")
                cur.execute("USE SCHEMA SUFFOLK_ANALYTICS_SCHEMA;")
                print("Snowflake warehouse, database, and schema SUFFOLK_ANALYTICS_SCHEMA ensured.")
                
                # Create or replace file format for CSV.
                cur.execute("""
                    CREATE OR REPLACE FILE FORMAT REAL_ESTATE_CSV_FORMAT
                    TYPE = 'CSV'
                    FIELD_OPTIONALLY_ENCLOSED_BY = '\"'
                    SKIP_HEADER = 1;
                """)
                print("File format REAL_ESTATE_CSV_FORMAT created.")
                
                # Create or replace an external stage pointing to the S3 Merged_Files folder using inline AWS credentials.
                cur.execute(f"""
                    CREATE OR REPLACE STAGE REAL_ESTATE_STAGE
                    URL = 's3://{s3_bucket}/{s3_merged_folder}/'
                    CREDENTIALS=(AWS_KEY_ID='{os.getenv("AWS_ACCESS_KEY_ID")}', AWS_SECRET_KEY='{os.getenv("AWS_SECRET_ACCESS_KEY")}')
                    FILE_FORMAT = (FORMAT_NAME = 'REAL_ESTATE_CSV_FORMAT');
                """)
                print("External stage REAL_ESTATE_STAGE created.")
                
                # List files in the stage to ensure that CSV files exist.
                cur.execute("LIST @REAL_ESTATE_STAGE PATTERN='.*\\.csv';")
                stage_list = cur.fetchall()
                if not stage_list:
                    raise Exception("No CSV files found in the stage 'REAL_ESTATE_STAGE'. Please verify that merged CSV file(s) exist in the S3 folder.")
                else:
                    print("Files found in stage:")
                    for record in stage_list:
                        print(record[0])
                
                # Create or replace the target table with a fixed definition.
                create_table_sql = """
                    CREATE OR REPLACE TABLE MERGED_HOMEVALUES (
                        "RegionID" VARCHAR,
                        "RegionName" VARCHAR,
                        "StateName" VARCHAR,
                        "City" VARCHAR,
                        "CountyName" VARCHAR,
                        "Date" DATE,
                        "1Bed_HomeValue" VARCHAR,
                        "2Bed_HomeValue" VARCHAR,
                        "3Bed_HomeValue" VARCHAR,
                        "4Bed_HomeValue" VARCHAR,
                        "5Bed_HomeValue" VARCHAR,
                        "MA_Condo_HomeValues" VARCHAR,
                        "MA_Single_Family_HomeValues" VARCHAR,
                        "MA_SFRCondo_HomeValues" VARCHAR
                    );
                """
                cur.execute(create_table_sql)
                print("Target table MERGED_HOMEVALUES created with column names preserved.")
                
                # Use COPY command to load all CSV files from the external stage.
                copy_sql = """
                    COPY INTO MERGED_HOMEVALUES
                    FROM @REAL_ESTATE_STAGE
                    PATTERN='.*\\.csv'
                    ON_ERROR = 'ABORT_STATEMENT';
                """
                cur.execute(copy_sql)
                print("Data loaded into Snowflake table MERGED_HOMEVALUES successfully.")
                conn.commit()
    except Exception as e:
        print(f"Snowflake pipeline error: {e}")

if __name__ == "__main__":
    snowflake_pipeline()
