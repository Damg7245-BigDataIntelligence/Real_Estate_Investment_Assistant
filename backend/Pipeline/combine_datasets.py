import pandas as pd
import pathlib

def combine_datasets():
    # Read both CSV files
    historical_df = pd.read_csv('datasets/boston_metro_zip_historical.csv')
    current_df = pd.read_csv('datasets/boston_metro_zip_details_cleaned.csv')
    
    # Add year 2023 to current data
    current_df['Year'] = 2023
    
    # Get common columns between both dataframes
    common_columns = list(set(historical_df.columns) & set(current_df.columns))
    
    # Create a combined dataframe with all years
    combined_df = pd.concat([
        historical_df[common_columns],
        current_df[common_columns]
    ], ignore_index=True)
    
    # Sort by ZipCode and Year
    combined_df = combined_df.sort_values(['ZipCode', 'Year'])
    
    # Save the combined dataset
    combined_df.to_csv('datasets/boston_metro_zip_combined.csv', index=False)
    print("✅ Combined dataset created successfully!")

if __name__ == "__main__":
    combine_datasets() 