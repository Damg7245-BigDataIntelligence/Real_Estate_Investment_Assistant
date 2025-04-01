import os
from dotenv import load_dotenv
import snowflake.connector

# Load environment variables from .env file
load_dotenv()

# Snowflake connection configuration (using your provided details)
conn_config = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "role": os.getenv("SNOWFLAKE_ROLE")
    # Optionally include warehouse if desired:
    # "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "REAL_ESTATE_WH")
}

# AWS/S3 details from .env
s3_bucket = os.getenv("AWS_S3_BUCKET_NAME")  # Expected to be "real-estate-data-ma"
s3_folder = "current_listings"              # Folder in the bucket

# Names for warehouse, database, and schema
warehouse_name = "REAL_ESTATE_WH"
database_name = "REAL_ESTATE_DB"
schema_name = "CURRENT_LISTINGS"

# Target table name in the CURRENT_LISTINGS schema
table_name = "LISTINGS"

try:
    with snowflake.connector.connect(**conn_config) as conn:
        with conn.cursor() as cur:
            # Create Warehouse if not exists
            cur.execute(f"""
                CREATE WAREHOUSE IF NOT EXISTS {warehouse_name}
                WAREHOUSE_SIZE = 'SMALL'
                AUTO_SUSPEND = 60
                AUTO_RESUME = TRUE;
            """)
            print(f"Warehouse {warehouse_name} ensured.")

            # Create Database if not exists
            cur.execute(f"CREATE DATABASE IF NOT EXISTS {database_name};")
            print(f"Database {database_name} ensured.")

            # Set database and create the target schema CURRENT_LISTINGS
            cur.execute(f"USE DATABASE {database_name};")
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name};")
            cur.execute(f"USE SCHEMA {database_name}.{schema_name};")
            print(f"Schema {schema_name} ensured.")

            # Create or replace a file format for CSV files
            cur.execute("""
                CREATE OR REPLACE FILE FORMAT REAL_ESTATE_CSV_FORMAT
                TYPE = 'CSV'
                FIELD_OPTIONALLY_ENCLOSED_BY = '\"'
                SKIP_HEADER = 1;
            """)
            print("File format REAL_ESTATE_CSV_FORMAT created.")

            # Create an external stage using direct AWS credentials
            cur.execute(f"""
                CREATE OR REPLACE STAGE {schema_name}.MY_STAGE
                URL = 's3://{s3_bucket}/{s3_folder}/'
                CREDENTIALS=(AWS_KEY_ID='{os.getenv("AWS_ACCESS_KEY_ID")}' AWS_SECRET_KEY='{os.getenv("AWS_SECRET_ACCESS_KEY")}')
                FILE_FORMAT = (FORMAT_NAME = 'REAL_ESTATE_CSV_FORMAT');
            """)
            print(f"External stage {schema_name}.MY_STAGE created using direct credentials.")

            # Create or replace the target table with quoted identifiers to preserve the case.
            cur.execute(f"""
                CREATE OR REPLACE TABLE {schema_name}.{table_name} (
                    "City" VARCHAR,
                    "Address" VARCHAR,
                    "StateName" VARCHAR,
                    "RegionName" VARCHAR,
                    "Price" VARCHAR,
                    "Beds" VARCHAR,
                    "Baths" VARCHAR,
                    "Sqft" VARCHAR,
                    "Url" VARCHAR,
                    "Date" DATE
                );
            """)
            print(f"Table {schema_name}.{table_name} created with column names preserved.")

            # Load data from S3 stage into the target table.
            cur.execute(f"""
                COPY INTO {schema_name}.{table_name}
                FROM @{schema_name}.MY_STAGE
                FILE_FORMAT = (FORMAT_NAME = 'REAL_ESTATE_CSV_FORMAT')
                PATTERN='.*\\.csv'
                ON_ERROR = 'ABORT_STATEMENT';
            """)
            print("Data loaded into table successfully.")

            # Update the RegionName column to strip any leading zeros.
            cur.execute(f"""
                UPDATE {schema_name}.{table_name}
                SET "RegionName" = REGEXP_REPLACE("RegionName", '^0+', '');
            """)
            print("RegionName column updated to remove any leading zeros.")

            conn.commit()
            print("Snowflake pipeline completed successfully.")

except Exception as e:
    print("Snowflake pipeline error:", e)
