import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def create_snowflake_table():
    """Create Snowflake table and load data from CSV"""
    # Snowflake connection parameters
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        role=os.getenv('SNOWFLAKE_ROLE')  # Added role parameter
    )
    
    try:
        cursor = conn.cursor()
        
        # Create Warehouse (if it doesn't exist)
        cursor.execute("""
            CREATE WAREHOUSE IF NOT EXISTS BOSTON_REAL_ESTATE_WH
            WAREHOUSE_SIZE = 'SMALL'
            AUTO_SUSPEND = 60
            AUTO_RESUME = TRUE;
        """)
        
        # Create Database (if it doesn't exist)
        cursor.execute("""
            CREATE DATABASE IF NOT EXISTS SUFFOLK_REAL_ESTATE_DB;
        """)
        
        # Use the database
        cursor.execute("USE DATABASE SUFFOLK_REAL_ESTATE_DB;")
        
        # Create Schema (if it doesn't exist)
        cursor.execute("""
            CREATE SCHEMA IF NOT EXISTS SUFFOLK_REAL_ESTATE_SCHEMA;
        """)
        
        # Use the schema
        cursor.execute("USE SCHEMA SUFFOLK_REAL_ESTATE_DB.SUFFOLK_REAL_ESTATE_SCHEMA;")
        
        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SUFFOLK_REAL_ESTATE_DEMOGRAPHICS (
                ZIP_CODE VARCHAR,
                YEAR INTEGER,
                TOTAL_POPULATION INTEGER,
                WHITE_POP INTEGER,
                BLACK_POP INTEGER,
                ASIAN_POP INTEGER,
                MALE_POP INTEGER,
                FEMALE_POP INTEGER,
                TOTAL_HOUSING_UNITS INTEGER,
                AVERAGE_HOUSE_VALUE DECIMAL(15,2),
                INCOME_PER_HOUSEHOLD DECIMAL(15,2),
                PERSONS_PER_HOUSEHOLD DECIMAL(5,2),
                CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        
        # Read CSV data
        df = pd.read_csv('datasets/suffolk_metro_zip_combined.csv')
        
        # Convert numeric columns to appropriate types and handle NaN values
        numeric_columns = [
            'Total_Population', 'WhitePop', 'BlackPop', 'AsianPop',
            'MalePop', 'FemalePop', 'TotalHousingUnits',
            'AverageHouseValue', 'IncomePerHousehold', 'PersonsPerHousehold'
        ]
        
        for col in numeric_columns:
            # Convert to numeric and replace NaN with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Prepare data for insertion
        for _, row in df.iterrows():
            try:
                # Convert values to Python native types, handling NaN and 0.0 values
                values = (
                    str(row['ZipCode']),
                    int(row['Year']),
                    int(row['Total_Population']),
                    int(row['WhitePop']),
                    int(row['BlackPop']),
                    int(row['AsianPop']),
                    int(row['MalePop']),
                    int(row['FemalePop']),
                    int(row['TotalHousingUnits']),
                    float(row['AverageHouseValue']) if pd.notna(row['AverageHouseValue']) else 0.0,
                    float(row['IncomePerHousehold']) if pd.notna(row['IncomePerHousehold']) else 0.0,
                    float(row['PersonsPerHousehold']) if pd.notna(row['PersonsPerHousehold']) else 0.0
                )
                
                cursor.execute("""
                    INSERT INTO SUFFOLK_REAL_ESTATE_DEMOGRAPHICS (
                        ZIP_CODE, YEAR, TOTAL_POPULATION, WHITE_POP, BLACK_POP,
                        ASIAN_POP, MALE_POP, FEMALE_POP, TOTAL_HOUSING_UNITS,
                        AVERAGE_HOUSE_VALUE, INCOME_PER_HOUSEHOLD, PERSONS_PER_HOUSEHOLD
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, values)
            except Exception as e:
                print(f"Error processing row: {row['ZipCode']}, Year: {row['Year']}")
                print(f"Error details: {str(e)}")
                continue
        
        conn.commit()
        print("✅ Data successfully loaded into Snowflake")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_snowflake_table() 