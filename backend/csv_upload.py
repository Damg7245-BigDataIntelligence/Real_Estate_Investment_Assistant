import os
import pandas as pd
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

##################################
# Part 1: CSV Processing & S3 Upload
##################################

# S3 configuration
s3_bucket = os.getenv("AWS_S3_BUCKET_NAME")
if not s3_bucket:
    raise ValueError("AWS_S3_BUCKET_NAME is not set in the .env file.")

s3_filtered_folder = "Filtered_MA_Data"   # Folder for processed (filtered) files on S3
s3_merged_folder = "Merged_Files"          # Separate folder for merged file on S3
s3_original_folder = "Original_Files"      # Folder for original files on S3

s3_client = boto3.client('s3')

# Local folders
input_folder = "backend/Real_Estate_data"  # Folder with raw CSV files
filtered_folder = os.path.join("backend", "Filtered_MA_Data")
os.makedirs(filtered_folder, exist_ok=True)

# List of CSV files and desired output names.
files = [
    ("Zip_zhvi_bdrmcnt_1_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_1Bed.csv"),
    ("Zip_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_2Bed.csv"),
    ("Zip_zhvi_bdrmcnt_3_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_3Bed.csv"),
    ("Zip_zhvi_bdrmcnt_4_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_4Bed.csv"),
    ("Zip_zhvi_bdrmcnt_5_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_HomeValues_5Bed.csv"),
    ("Zip_zhvi_uc_condo_tier_0.33_0.67_sm_sa_month.csv", "MA_Condo_HomeValues.csv"),
    ("Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv", "MA_Single_Family_HomeValues.csv"),
    ("Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "MA_SFRCondo_HomeValues.csv"),
]

# Desired cities for filtering
desired_cities = ["Boston", "Dorchester", "Revere", "Chelsea"]

# Target years for December data
target_years = {2020, 2021, 2022, 2023, 2024}

def s3_key_exists(bucket, key):
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

def upload_to_s3(file_path, bucket, key):
    """Upload a local file to S3."""
    try:
        s3_client.upload_file(file_path, bucket, key)
        print(f"✅ Uploaded {file_path} to s3://{bucket}/{key}")
    except Exception as e:
        print(f"❌ Failed to upload {file_path} to S3: {e}")

def parse_date_header(date_str):
    """
    Attempts to parse a date header.
    If the 5th character is '-', assumes '%Y-%m-%d';
    otherwise, if the 3rd character is '-', assumes '%d-%m-%Y'.
    """
    date_str = date_str.strip()
    try:
        if len(date_str) >= 10:
            if date_str[4] == '-':
                return datetime.strptime(date_str, "%Y-%m-%d")
            elif date_str[2] == '-':
                return datetime.strptime(date_str, "%d-%m-%Y")
    except Exception:
        return None
    return None

def is_target_dec31(date_str):
    """
    Returns True if date_str represents December 31 of one of the target years.
    """
    dt = parse_date_header(date_str)
    if dt is None:
        return False
    return (dt.day == 31) and (dt.month == 12) and (dt.year in target_years)

def process_file(file_name, output_name):
    """
    Processes a single CSV file:
      - Uploads the original file to S3 if not already present.
      - Reads the CSV and filters for rows where StateName=="MA" and City is in desired_cities.
      - Identifies candidate date columns (all columns not in common columns).
      - Selects only those columns that represent December 31 of a target year.
      - Melts the DataFrame using these date columns.
      - Renames the melted value column (e.g. "HomeValue") based on output_name.
      - Saves the processed file locally and uploads it to S3.
    """
    original_key = f"{s3_original_folder}/{file_name}"
    filtered_key = f"{s3_filtered_folder}/{output_name}"
    
    # Upload original file if not exists.
    if not s3_key_exists(s3_bucket, original_key):
        input_path = os.path.join(input_folder, file_name)
        try:
            with open(input_path, "rb") as f:
                s3_client.put_object(Body=f.read(), Bucket=s3_bucket, Key=original_key)
            print(f"✅ Original file {file_name} uploaded to s3://{s3_bucket}/{original_key}")
        except Exception as e:
            print(f"❌ Error uploading original file {file_name}: {e}")
    else:
        print(f"ℹ️ Original file {original_key} already exists on S3.")
    
    # Read and filter the CSV.
    input_path = os.path.join(input_folder, file_name)
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"❌ Error reading {file_name}: {e}")
        return None

    if "StateName" not in df.columns or "City" not in df.columns:
        print(f"❌ Required columns not found in {file_name}")
        return None
    df = df[(df["StateName"] == "MA") & (df["City"].isin(desired_cities))]
    if df.empty:
        print(f"⚠️ No data for MA and desired cities in {file_name}")
        return None

    # Define common columns (exclude RegionType as requested).
    common_cols = ["RegionID", "RegionName", "StateName", "City", "CountyName"]
    common_cols = [col for col in common_cols if col in df.columns]
    
    # Candidate date columns: columns not in common_cols.
    candidate_date_cols = [col for col in df.columns if col not in common_cols]
    print(f"Candidate date columns in {file_name}: {candidate_date_cols}")
    
    # Keep only columns that represent December 31 of a target year.
    selected_date_cols = [col for col in candidate_date_cols if is_target_dec31(col)]
    if not selected_date_cols:
        print(f"⚠️ No December 31 columns for target years found in {file_name}")
        return None

    # Melt the DataFrame using the selected date columns.
    df_long = pd.melt(df, id_vars=common_cols, value_vars=selected_date_cols,
                      var_name="Date", value_name="HomeValue")
    if df_long.empty:
        print(f"⚠️ No data after melting in {file_name}")
        return None

    # Rename the HomeValue column based on output_name.
    new_home_col = output_name.replace("MA_HomeValues_", "").replace(".csv", "") + "_HomeValue"
    df_long = df_long.rename(columns={"HomeValue": new_home_col})
    
    # Reorder columns and sort.
    final_cols = common_cols + ["Date", new_home_col]
    df_long = df_long[final_cols]
    df_long["City"] = pd.Categorical(df_long["City"], categories=desired_cities, ordered=True)
    df_long = df_long.sort_values(by=["City", "Date"])
    
    # Save processed file locally.
    local_output_path = os.path.join(filtered_folder, output_name)
    try:
        df_long.to_csv(local_output_path, index=False)
        print(f"✅ Processed file saved locally as {local_output_path}")
    except Exception as e:
        print(f"❌ Error saving processed file {output_name}: {e}")
        return None

    # Upload processed file to S3.
    upload_to_s3(local_output_path, s3_bucket, filtered_key)
    return local_output_path

def process_all_files():
    """Processes all files and returns a list of local paths for processed CSV files."""
    processed_files = []
    for file_name, output_name in files:
        print(f"\nProcessing file: {file_name}")
        processed = process_file(file_name, output_name)
        if processed:
            processed_files.append(processed)
    return processed_files

def merge_processed_files(processed_file_paths):
    """
    Merges all processed CSV files on common keys:
    RegionID, RegionName, StateName, City, CountyName, Date.
    Each file contributes its own home value column.
    """
    merged_df = None
    common_keys = ["RegionID", "RegionName", "StateName", "City", "CountyName", "Date"]
    for file_path in processed_file_paths:
        try:
            df = pd.read_csv(file_path)
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=common_keys, how="outer")
            print(f"File {file_path} merged successfully.")
        except Exception as e:
            print(f"❌ Error merging file {file_path}: {e}")
    if merged_df is not None:
        merged_df["City"] = pd.Categorical(merged_df["City"], categories=desired_cities, ordered=True)
        merged_df = merged_df.sort_values(by=["City", "Date"])
    return merged_df

def part1_pipeline():
    """
    Runs the CSV processing pipeline:
      - Processes all files,
      - Merges them,
      - Saves the merged file locally in a "Merged" folder,
      - Uploads the merged file to S3 in a separate folder (e.g., "Merged_Files").
    """
    processed_files = process_all_files()
    if not processed_files:
        print("❌ No processed files available.")
        return None
    merged_df = merge_processed_files(processed_files)
    if merged_df is None or merged_df.empty:
        print("❌ Merged DataFrame is empty.")
        return None
    merged_filename = "Merged_MA_HomeValues.csv"
    merged_folder = os.path.join("Merged")
    os.makedirs(merged_folder, exist_ok=True)
    merged_filepath = os.path.join(merged_folder, merged_filename)
    try:
        merged_df.to_csv(merged_filepath, index=False)
        print(f"✅ Merged CSV saved locally as {merged_filepath}")
    except Exception as e:
        print(f"❌ Error saving merged CSV: {e}")
        return None
    # Upload merged file to S3 in a separate folder
    merged_s3_key = f"Merged_Files/{merged_filename}"
    upload_to_s3(merged_filepath, s3_bucket, merged_s3_key)
    return merged_filepath

if __name__ == "__main__":
    merged_file = part1_pipeline()
