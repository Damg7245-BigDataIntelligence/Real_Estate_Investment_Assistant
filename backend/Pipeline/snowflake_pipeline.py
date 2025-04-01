import os
import io
import pandas as pd
import boto3
import snowflake.connector
from io import StringIO
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Load environment variables from .env file
load_dotenv()

##################################
# Part 1: Process & Upload to S3 #
##################################

# S3 configuration
s3_bucket = os.getenv("AWS_S3_BUCKET_NAME")
if not s3_bucket:
    raise ValueError("AWS_S3_BUCKET_NAME is not set in the .env file.")

# S3 folders for filtered data and original files
s3_filtered_folder = "Filtered_MA_Data"
s3_original_folder = "Original_Files"

s3_client = boto3.client('s3')

# Local folder where the raw CSV files are stored
input_folder = "Real_Estate_data"

# List of CSV files along with their descriptive output names
files = [
    ("Zip_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_1Bed.csv"),
    ("Zip_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_2Bed.csv"),
    ("Zip_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_3Bed.csv"),
    ("Zip_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_4Bed.csv"),
    ("Zip_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_5Bed.csv"),
    ("Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv", "MA_Condo_HomeValues.csv"),
    ("Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv", "MA_SFR_HomeValues.csv"),
    ("Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_SFRCondo_HomeValues.csv"),
    ("Metro_invt_fs_uc_sfrcondo_sm_month.csv", "MA_Metro_Inventory_Monthly.csv"),
    ("Metro_invt_fs_uc_sfrcondo_sm_week.csv", "MA_Metro_Inventory_Weekly.csv"),
    ("Metro_mlp_uc_sfrcondo_sm_month.csv", "MA_Metro_MedianListPrice_Monthly.csv"),
    ("Metro_new_listings_uc_sfrcondo_sm_month.csv", "MA_Metro_NewListings_Monthly.csv"),
    ("Metro_new_listings_uc_sfrcondo_sm_week.csv", "MA_Metro_NewListings_Weekly.csv"),
    ("Metro_new_pending_uc_sfrcondo_sm_month.csv", "MA_Metro_NewPending_Monthly.csv"),
    ("Metro_new_pending_uc_sfrcondo_sm_week.csv", "MA_Metro_NewPending_Weekly.csv"),
    ("Metro_zhvi_uc_sfrcondo_tier_0.0_0.33_sm_sa_month.csv", "MA_Metro_ZHVI_BottomTier.csv"),
    ("Metro_zhvi_uc_sfrcondo_tier_0.67_1.0_sm_sa_month.csv", "MA_Metro_ZHVI_TopTier.csv")
]

def s3_key_exists(bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

def upload_to_s3(csv_data, bucket, key):
    """Uploads CSV data (as a string) to the specified S3 bucket and key."""
    try:
        s3_client.put_object(Body=csv_data, Bucket=bucket, Key=key)
        print(f"✅ Successfully uploaded to s3://{bucket}/{key}")
    except Exception as e:
        print(f"❌ Failed to upload {key} to S3: {e}")

def process_and_upload_files():
    """
    For each CSV file:
      - Upload the original file to the Original_Files folder if it doesn't exist.
      - Read the file, filter rows where StateName == "MA", and
        upload the resulting wide-format CSV to the Filtered_MA_Data folder if not already present.
    """
    for file_name, output_name in files:
        # Define S3 keys for original and filtered files
        original_key = f"{s3_original_folder}/{file_name}"
        filtered_key = f"{s3_filtered_folder}/{output_name}"
        
        # Upload original file if not present
        if not s3_key_exists(s3_bucket, original_key):
            input_path = os.path.join(input_folder, file_name)
            try:
                with open(input_path, "rb") as f:
                    s3_client.put_object(Body=f.read(), Bucket=s3_bucket, Key=original_key)
                print(f"✅ Original file uploaded to s3://{s3_bucket}/{original_key}")
            except Exception as e:
                print(f"❌ Error uploading original file {file_name}: {e}")
        else:
            print(f"ℹ️ Original file {original_key} already exists in S3. Skipping upload.")
        
        # Check if filtered file exists
        if s3_key_exists(s3_bucket, filtered_key):
            print(f"ℹ️ Filtered file {filtered_key} already exists in S3. Skipping upload.")
            continue
        
        # Process filtered file upload
        input_path = os.path.join(input_folder, file_name)
        try:
            df = pd.read_csv(input_path)
            if "StateName" in df.columns:
                df_ma = df[df["StateName"] == "MA"]
                if df_ma.empty:
                    print(f"⚠️ No MA data found in {file_name}")
                    continue
                # Keep the wide-format for later unpivoting.
                csv_buffer = StringIO()
                df_ma.to_csv(csv_buffer, index=False)
                csv_content = csv_buffer.getvalue()
                upload_to_s3(csv_content, s3_bucket, filtered_key)
            else:
                print(f"❌ 'StateName' column not found in {file_name}")
        except Exception as e:
            print(f"❌ Error processing {file_name}: {e}")
    print("\n🎉 All files processed and uploaded to S3.")

####################################
# Part 2: Snowflake Ingestion Logic #
####################################

def get_table_definitions(local_file):
    """
    Reads the local CSV file (filtered for MA) and dynamically determines:
      - The SQL definition for the staging table (all columns).
      - The comma-separated list of date columns for UNPIVOT.
      - The list of non-date columns actually present.
    Returns a tuple: (staging_columns_sql, unpivot_list_sql, actual_non_date_cols)
    """
    # Expected non-date columns (adjust as needed)
    expected_non_date_cols = ["RegionID", "SizeRank", "RegionName", "RegionType", 
                              "StateName", "State", "City", "Metro", "CountyName"]
    try:
        df = pd.read_csv(local_file)
        df_ma = df[df["StateName"] == "MA"]
        if df_ma.empty:
            print(f"⚠️ No MA data in {local_file}")
            return None, None, None
        cols = list(df_ma.columns)
        # Determine actual non-date columns present
        actual_non_date_cols = [col for col in expected_non_date_cols if col in cols]
        # Treat all other columns as date columns
        date_cols = [col for col in cols if col not in actual_non_date_cols]
        
        # Build staging table column definitions: non-date columns as STRING, date columns as FLOAT.
        staging_cols = []
        for col in cols:
            if col in actual_non_date_cols:
                staging_cols.append(f'"{col}" STRING')
            else:
                staging_cols.append(f'"{col}" FLOAT')
        staging_columns_sql = ",\n    ".join(staging_cols)
        unpivot_list_sql = ", ".join(f'"{col}"' for col in date_cols)
        return staging_columns_sql, unpivot_list_sql, actual_non_date_cols
    except Exception as e:
        print(f"❌ Error generating table definitions from {local_file}: {e}")
        return None, None, None

def table_exists(cur, table_name):
    """Checks if a table exists in the current Snowflake schema."""
    cur.execute(f"SHOW TABLES LIKE '{table_name}'")
    results = cur.fetchall()
    return len(results) > 0

def snowflake_pipeline():
    """
    Connects to Snowflake and, for each filtered CSV file uploaded to S3:
      - Checks if the normalized table already exists. If so, skip processing that file.
      - Otherwise, dynamically creates a staging table based on the CSV schema.
      - Loads the CSV from S3 into the staging table.
      - Creates a normalized table by unpivoting all date columns.
        The header of each date column (sales date) is converted to a DATE.
    """
    # Build Snowflake connection configuration
    conn_config = {
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "role": os.getenv("SNOWFLAKE_ROLE")
    }
    
    try:
        with snowflake.connector.connect(**conn_config) as conn:
            with conn.cursor() as cur:
                # Create Warehouse, Database, and Schema if they don't exist
                cur.execute("""
                    CREATE WAREHOUSE IF NOT EXISTS REAL_ESTATE_WH
                    WAREHOUSE_SIZE = 'SMALL'
                    AUTO_SUSPEND = 60
                    AUTO_RESUME = TRUE;
                """)
                cur.execute("CREATE DATABASE IF NOT EXISTS REAL_ESTATE_DB;")
                cur.execute("USE DATABASE REAL_ESTATE_DB;")
                cur.execute("CREATE SCHEMA IF NOT EXISTS REAL_ESTATE_SCHEMA;")
                cur.execute("USE SCHEMA REAL_ESTATE_DB.REAL_ESTATE_SCHEMA;");
                
                # Create a file format for CSV files
                cur.execute("""
                    CREATE OR REPLACE FILE FORMAT REAL_ESTATE_CSV_FORMAT
                    TYPE = 'CSV'
                    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                    SKIP_HEADER = 1;
                """)
                
                # Create a stage that points to your S3 filtered folder
                cur.execute(f"""
                    CREATE OR REPLACE STAGE REAL_ESTATE_STAGE
                    URL = 's3://{s3_bucket}/{s3_filtered_folder}/'
                    STORAGE_INTEGRATION = real_estate_integration
                    FILE_FORMAT = (FORMAT_NAME = 'REAL_ESTATE_CSV_FORMAT');
                """)
                
                # Process each file in the files list
                for file_name, output_name in files:
                    base_name = output_name.replace(".csv", "").upper()
                    norm_table = base_name + "_NORM"
                    
                    # Check if the normalized table already exists; if yes, skip processing this file.
                    if table_exists(cur, norm_table):
                        print(f"ℹ️ Normalized table {norm_table} already exists. Skipping ingestion for {output_name}.")
                        continue
                    
                    stg_table = base_name + "_STG"
                    
                    local_file_path = os.path.join(input_folder, file_name)
                    try:
                        df = pd.read_csv(local_file_path)
                        df_ma = df[df["StateName"] == "MA"]
                        if df_ma.empty:
                            print(f"⚠️ No MA data in {file_name}")
                            continue
                        # Save filtered file temporarily to determine schema
                        temp_file = f"temp_{output_name}"
                        df_ma.to_csv(temp_file, index=False)
                    except Exception as e:
                        print(f"❌ Error reading {file_name}: {e}")
                        continue
                    
                    staging_columns_sql, unpivot_list_sql, actual_non_date_cols = get_table_definitions(temp_file)
                    os.remove(temp_file)
                    
                    if not staging_columns_sql or not unpivot_list_sql:
                        print(f"❌ Could not generate table definitions for {output_name}")
                        continue
                    
                    # Create the staging table dynamically
                    create_stg_sql = f"""
                        CREATE OR REPLACE TABLE {stg_table} (
                            {staging_columns_sql}
                        );
                    """
                    cur.execute(create_stg_sql)
                    
                    # Load data into the staging table from S3 using the FILES option
                    cur.execute(f"""
                        COPY INTO {stg_table}
                        FROM @REAL_ESTATE_STAGE
                        FILES = ('{output_name}')
                        FILE_FORMAT = (FORMAT_NAME = 'REAL_ESTATE_CSV_FORMAT')
                        ON_ERROR = 'ABORT_STATEMENT';
                    """)
                    
                    # Build non-date column selection for the normalized table
                    non_date_select = ", ".join(f'"{col}"' for col in actual_non_date_cols)
                    
                    # Create the normalized table by unpivoting all date columns.
                    # We assume the date headers are in the format 'DD-MM-YYYY'.
                    create_norm_sql = f"""
                        CREATE OR REPLACE TABLE {norm_table} AS
                        SELECT 
                            {non_date_select},
                            TO_DATE(DateStr, 'DD-MM-YYYY') AS Date,
                            HomeValue
                        FROM {stg_table}
                        UNPIVOT (
                            HomeValue FOR DateStr IN ({unpivot_list_sql})
                        );
                    """
                    cur.execute(create_norm_sql)
                    print(f"✅ Normalized table {norm_table} created from {output_name}.")
                
                conn.commit()
                print("✅ Snowflake pipeline completed: All staging and normalized tables created.")
    except Exception as e:
        print(f"❌ Snowflake pipeline error: {e}")

#########################
# Main Pipeline Control #
#########################

if __name__ == "__main__":
    process_and_upload_files()
    snowflake_pipeline()
