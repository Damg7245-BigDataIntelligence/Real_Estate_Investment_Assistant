import boto3
import os
from datetime import datetime
import uuid
from dotenv import load_dotenv

load_dotenv()

def get_s3_client():
    """Get configured S3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )

def upload_file_to_s3(file_content, filename, folder="default", is_temp=False):
    """
    Upload a file to S3 and return the key.
    
    Args:
        file_content: The content to upload
        filename: The filename to use
        folder: The folder to store in
        is_temp: Whether this is a temporary file that should expire
        
    Returns:
        s3_key: The S3 key where the file was stored
    """
    s3_client = get_s3_client()
    bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
    
    # For temporary visualizations, use the temp subfolder
    if is_temp and folder == "visualizations":
        s3_key = f"{folder}/temp/{filename}"
    else:
        s3_key = f"{folder}/{filename}"
    
    # Upload the file
    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
            Body=file_content
    )
    
    return s3_key

def generate_presigned_url(s3_key, expiry=3600):
    """Generate a presigned URL for an S3 object."""
    s3_client = get_s3_client()
    bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
    
    # Adjust expiry based on folder
    if "/visualizations/temp/" in s3_key:
        # Short expiry for temporary visualizations
        actual_expiry = min(7200, expiry)  # Max 2 hours
    else:
        actual_expiry = expiry
    
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={
            'Bucket': bucket_name,
            'Key': s3_key
        },
        ExpiresIn=actual_expiry
    )
    
    return url

def upload_visualization_to_s3(image_data, prefix, filename):
    """
    Upload visualization to S3 with organized folder structure.
    
    Args:
        image_data: Binary image data
        prefix: Folder path (e.g., 'visualizations/temp/query_20240315_123456')
        filename: Name of the file (e.g., 'time_series.png')
    """
    try:
        # Create the full S3 key
        s3_key = f"{prefix}/{filename}"
        
        # Upload to S3
        s3_client = boto3.client('s3')
        s3_client.put_object(
            Bucket=os.getenv('AWS_S3_BUCKET_NAME'),
            Key=s3_key,
            Body=image_data,
            ContentType='image/png'
        )
        
        # Generate presigned URL with 24-hour expiry
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': os.getenv('AWS_S3_BUCKET_NAME'),
                'Key': s3_key,
                'ResponseContentType': 'image/png'
            },
            ExpiresIn=86400  # 24 hours
        )
        
        return presigned_url
        
    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise

def setup_visualization_lifecycle_rule():
    """Setup lifecycle rule to expire temporary visualizations after 1 day."""
    print("Setting up visualization lifecycle rule")
    s3_client = get_s3_client()
    bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
    try:
        # Create lifecycle configuration
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'Expire-temp-visualizations',
                    'Status': 'Enabled',
                    'Filter': {
                        'Prefix': 'visualizations/temp/'
                    },
                    'Expiration': {
                        'Days': 1
                    }
                }
            ]
        }
        
        # Apply the configuration to the bucket
        s3_client.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_config
        )
        
        print(f"Lifecycle rule set for temporary visualizations in {bucket_name}")
        return True
    except Exception as e:
        print(f"Error setting lifecycle rule: {e}")
        return False