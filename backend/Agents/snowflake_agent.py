import os
import re
import io
import snowflake.connector
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import traceback
import sys
import json # Added for pretty printing dicts
import google.generativeai as genai
from backend.llm_response import generate_response_with_gemini

# --- Path Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

# Add project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- S3 Import / Dummy Function ---
try:
    from backend.s3_utils import upload_visualization_to_s3
except ImportError as e:
    # Fallback dummy function for local execution/testing
    def upload_visualization_to_s3(image_data, prefix, filename="visualization.png"):
        """Saves visualization locally instead of uploading to S3."""
        local_dir = os.path.join(script_dir, "temp_s3_uploads", prefix)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        try:
            with open(local_path, "wb") as f:
                f.write(image_data)
            abs_path = os.path.abspath(local_path)
            file_uri = f"file:///{abs_path.replace(os.sep, '/')}"
            return file_uri
        except Exception as write_e:
            print(f"Error saving visualization locally: {write_e}")
            return None

# --- Load .env ---
dotenv_path_project = os.path.join(project_root, '.env')
if os.path.exists(dotenv_path_project):
    load_dotenv(dotenv_path=dotenv_path_project)
else:
    load_dotenv()


# --- Snowflake Credentials ---
SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE")
SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SNOWFLAKE_SCHEMA = "SUFFOLK_ANALYTICS_SCHEMA"

# --- clean_sql Function ---
def clean_sql(raw_sql):
    """Removes comments, markdown backticks, and leading/trailing whitespace/semicolons."""
    if not raw_sql: return None
    # Remove SQL comments
    sql_no_comments = re.sub(r'--.*$', '', raw_sql, flags=re.MULTILINE)
    sql_no_comments = re.sub(r'/\*.*?\*/', '', sql_no_comments, flags=re.DOTALL)
    # Remove MySQL/Markdown style comments
    sql_no_comments = re.sub(r'##.*$', '', sql_no_comments, flags=re.MULTILINE)
    sql_no_comments = re.sub(r'#.*$', '', sql_no_comments, flags=re.MULTILINE)
    # Remove markdown code blocks
    cleaned = re.sub(r'^```sql\s*', '', sql_no_comments.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    # Strip leading/trailing whitespace and semicolons
    cleaned = cleaned.strip().rstrip(';')
    return cleaned if cleaned else None

# --- extract_and_clean_sql (Handles two queries) ---
def extract_and_clean_sql(gemini_response_text):
    """ Extracts TWO SQL queries from Gemini's response using '--- QUERY SEPARATOR ---'. """
    if not gemini_response_text:
        print("Error: Received empty response text from LLM.")
        return None, None

    # Check if it looks like we got a proper SQL response
    if "SELECT" not in gemini_response_text.upper() and "WITH" not in gemini_response_text.upper():
        print("Error: Response doesn't appear to contain SQL queries.")
        print("Response content:", gemini_response_text[:100] + "..." if len(gemini_response_text) > 100 else gemini_response_text)
        return None, None

    separator = '--- QUERY SEPARATOR ---'
    query1_raw = None
    query2_raw = None

    if separator in gemini_response_text:
        parts = gemini_response_text.split(separator, 1)
        query1_raw = parts[0]
        query2_raw = parts[1] if len(parts) > 1 else None
    else:
        # Fallback: Try to find SQL blocks if separator is missing
        sql_blocks = re.findall(r'```sql\s*(.*?)\s*```', gemini_response_text, re.DOTALL | re.IGNORECASE)
        if len(sql_blocks) >= 2:
            query1_raw = "```sql\n" + sql_blocks[0] + "\n```" # Add backticks for clean_sql
            query2_raw = "```sql\n" + sql_blocks[1] + "\n```"
        elif len(sql_blocks) == 1:
             query1_raw = "```sql\n" + sql_blocks[0] + "\n```"
        else:
            # Split by double newlines and find chunks starting with SELECT or WITH
            chunks = re.split(r'\n\s*\n', gemini_response_text)
            sql_chunks = []
            for chunk in chunks:
                chunk = chunk.strip()
                if chunk.upper().startswith("SELECT") or chunk.upper().startswith("WITH"):
                    sql_chunks.append(chunk)
            
            if len(sql_chunks) >= 2:
                query1_raw = sql_chunks[0]
                query2_raw = sql_chunks[1]
            elif len(sql_chunks) == 1:
                query1_raw = sql_chunks[0]
            else:
                # As a last resort, assume the whole response might be the first query if it looks like SQL
                temp_cleaned = clean_sql(gemini_response_text)
                if temp_cleaned and (temp_cleaned.upper().startswith("SELECT") or temp_cleaned.upper().startswith("WITH")):
                    query1_raw = gemini_response_text
                else:
                    return None, None

    query1 = clean_sql(query1_raw)
    query2 = clean_sql(query2_raw)

    # Validation
    if query1 and not (query1.upper().startswith("SELECT") or query1.upper().startswith("WITH")):
        query1 = None
    if query2 and not (query2.upper().startswith("SELECT") or query2.upper().startswith("WITH")):
        query2 = None

    # Print queries for debugging
    if query1:
        print("Query 1:", query1) 
    if query2:
        print("Query 2:", query2)

    return query1, query2

# --- fetch_snowflake_response (Query 2 compares City Averages for MULTIPLE metrics) ---
def fetch_snowflake_response(filter_dict):
    """
    Generates a prompt for generating TWO Snowflake SQL queries:
    1. Yearly average trends for a specific region within a city.
    2. Overall average values for multiple metrics across several cities (including the specified one).
    Uses the LLM response module to get the queries.
    """
    region = filter_dict.get("RegionName", "") # Specific region for query 1
    city = filter_dict.get("City", "")         # City filter for query 1 & base for query 2

    if not region or not city:
        print("Error: Missing 'RegionName' or 'City' in filter_dict for prompt generation.")
        return None

    # Define table name and schema 
    db_schema_table = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.MERGED_HOMEVALUES"

    # Prompt asking for BOTH queries clearly separated
    prompt = f"""
    Generate TWO Snowflake SQL queries separated by '--- QUERY SEPARATOR ---'.
    Provide ONLY the SQL code. Do NOT include ANY comments or explanations before, during, or after the queries.
    Do NOT include any markdown formatting like ```sql - only raw SQL.
    Use EXACT column names from the table, enclosed in double quotes (e.g., "RegionName").

    Table: {db_schema_table}

    ---

    QUERY 1: Specific Region Yearly Average Trend
    Objective: Calculate the yearly average home values for various metrics specifically for region '{region}' within the city '{city}'.
    Requirements:
    - SELECT the start of the year using DATE_TRUNC('YEAR', "Date") and alias it as "YEAR_START_DATE".
    - SELECT the average of "1Bed_HomeValue", "2Bed_HomeValue", "3Bed_HomeValue", and "MA_Single_Family_HomeValues", converting them to numbers using TRY_TO_NUMBER(). Alias them as "AVG_1BED", "AVG_2BED", "AVG_3BED", and "AVG_SFR" respectively. The AVG function inherently handles NULLs.
    - FROM the table {db_schema_table}.
    - WHERE "RegionName" equals '{region}' AND "City" equals '{city}'.
    - GROUP BY the calculated "YEAR_START_DATE".
    - ORDER BY "YEAR_START_DATE" in ascending order.
    - End the query with a semicolon.

    --- QUERY SEPARATOR ---

    QUERY 2: Cross-City Overall Average Value Comparison (Multi-Metric)
    Objective: Compare the OVERALL average values for '1Bed_HomeValue', '2Bed_HomeValue', '3Bed_HomeValue', and 'MA_Single_Family_HomeValues' across '{city}' and up to 4 other distinct sample cities found in the table.
    Requirements:
    - Use a Common Table Expression (CTE) named 'SampleCities' to select '{city}' plus up to 4 other distinct "City" values present in the table {db_schema_table}. Ensure the CTE selects distinct city names.
      Example CTE structure: WITH SampleCities AS (SELECT '{city}' AS "City" UNION SELECT DISTINCT "City" FROM {db_schema_table} WHERE "City" != '{city}' LIMIT 4)
    - In the main query after the CTE:
        - SELECT m."City".
        - SELECT the overall average of "1Bed_HomeValue", "2Bed_HomeValue", "3Bed_HomeValue", and "MA_Single_Family_HomeValues" using AVG(TRY_TO_NUMBER(...)). Alias these as "OVERALL_AVG_1BED", "OVERALL_AVG_2BED", "OVERALL_AVG_3BED", and "OVERALL_AVG_SFR" respectively.
        - FROM {db_schema_table} AS m.
        - INNER JOIN the SampleCities CTE AS s ON m."City" = s."City".
        - GROUP BY m."City".
        - ORDER BY "OVERALL_AVG_SFR" DESC (ordering by the average Single Family Residence value is a reasonable default for visualization).
    - End the query with a semicolon.
    """

    # Use generate_response_with_gemini instead of direct Gemini API call
    try:
        print("Generating SQL queries with LLM...")
        gemini_text = generate_response_with_gemini(
            query="Generate SQL for real estate analysis",
            context=prompt
        )
        
        if not gemini_text or len(gemini_text.strip()) < 20:
            print("Error: LLM returned empty or very short response")
            return None
            
        return gemini_text.strip()
    except Exception as e:
        print(f"Error generating SQL queries: {e}")
        return None

# --- fetch_snowflake_df (Robust version) ---
def fetch_snowflake_df(query, query_description="Query"):
    """Executes SQL query and returns DataFrame with proper type conversion."""
    conn = None
    cur = None

    try:
        print(f"Executing {query_description}...")
        print("SQL:", query)
        
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
            role=SNOWFLAKE_ROLE,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            connect_timeout=60,
            network_timeout=120
        )
        cur = conn.cursor()

        # Execute the query
        cur.execute(query)
        
        # Add debug prints
        print(f"Query results before DataFrame conversion:")
        print(f"Column names: {[col[0] for col in cur.description]}")
        print(f"First few rows: {cur.fetchmany(3)}")
        
        # Create DataFrame with explicit dtypes
        results = cur.fetchall()
        column_names = [col[0] for col in cur.description]
        
        df = pd.DataFrame(results, columns=column_names)
        
        # Convert numeric columns
        for col in df.columns:
            if any(term in col.upper() for term in ['VALUE', 'AVG', 'PRICE']):
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Convert date columns
        for col in df.columns:
            if 'DATE' in col.upper():
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
        return df

    except Exception as e:
        # General errors (network issues, unexpected problems)
        print(f"Error during Snowflake operation: {e}")
        return None # Indicate failure
    finally:
        # Ensure cursor and connection are closed
        if cur:
            try: cur.close()
            except Exception: pass
        if conn:
            try: conn.close()
            except Exception: pass

# --- create_and_save_graph (Handles specific region trend AND cross-city GROUPED BAR chart) ---
def create_and_save_graph(df, query_type, timestamp, metadata_filters=None):
    """
    Creates and saves visualizations based on the query type and DataFrame.

    Args:
        df (pd.DataFrame): The data to visualize.
        query_type (str): 'single_region_trend' or 'cross_city_comparison'.
        timestamp (str): Timestamp string for unique filenames.
        metadata_filters (dict, optional): Filters used for context in titles.

    Returns:
        list: A list of dictionaries, each containing info about a saved visualization (url, type, title, columns). Returns empty list on failure or no data.
    """
    if df is None or df.empty:
        return []

    visualizations = []
    viz_folder = f"visualizations/query_{timestamp}" # Relative path for S3 prefix/local dir

    # Standardize column names to UPPERCASE for reliable access
    original_columns = df.columns.tolist()
    df_plot = df.copy() # Avoid modifying the original DataFrame passed to the function
    df_plot.columns = [str(col).upper() for col in df_plot.columns]

    # Extract filter context for titles
    city_filter = metadata_filters.get('City', 'N/A') if metadata_filters else 'N/A'
    region_filter = metadata_filters.get('RegionName', 'N/A') if metadata_filters else 'N/A'
    filters_str_region = f"(City: {city_filter}, Region: {region_filter})"
    filters_str_city = f"(City: {city_filter} & Others)"

    plt.style.use('seaborn-v0_8-darkgrid') # Use a visually appealing style

    try:
        # === GRAPH 1: Single Region Yearly Time Series ===
        if query_type == 'single_region_trend':
            # Identify potential date/time column (assuming only one)
            date_col = next((c for c in df_plot.columns if 'DATE' in c), None)
            # Identify potential average value columns
            numeric_cols = sorted([c for c in df_plot.columns if c.startswith('AVG_')]) # Sort for consistent plot order

            if not date_col or not numeric_cols:
                return []

            # Ensure date column is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df_plot[date_col]):
                df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors='coerce')

            # Ensure numeric columns are numeric type
            valid_numeric_cols = []
            for col in numeric_cols:
                if not pd.api.types.is_numeric_dtype(df_plot[col]):
                    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
                # Check if conversion resulted in all NaNs which means the original data was bad
                if not df_plot[col].isnull().all():
                    valid_numeric_cols.append(col)

            if not valid_numeric_cols:
                return []

            # Drop rows where date is invalid (NaT) or *all* valid numeric columns are NaN
            df_plot_cleaned = df_plot.dropna(subset=[date_col], how='any')
            df_plot_cleaned = df_plot_cleaned.dropna(subset=valid_numeric_cols, how='all') # Keep row if at least one value exists
            df_plot_cleaned = df_plot_cleaned.sort_values(by=date_col) # Ensure data is sorted by date

            if df_plot_cleaned.empty:
                return []

            plt.figure(figsize=(12, 6))
            plot_title = f"Yearly Average Home Value Trend {filters_str_region}"
            ylabel = "Average Value ($)"
            count_plotted = 0

            for col in valid_numeric_cols:
                # Only plot if there's actual data for this column after cleaning
                if not df_plot_cleaned[col].isnull().all():
                    label = col.replace('AVG_', '').replace('_', ' ').replace('SFR', 'SFR').title()
                    plt.plot(df_plot_cleaned[date_col], df_plot_cleaned[col], label=label, marker='o', linestyle='-')
                    count_plotted += 1

            if count_plotted > 0:
                plt.xlabel('Year')
                plt.ylabel(ylabel)
                plt.title(plot_title, fontsize=14, fontweight='bold')
                plt.legend(title="Metric Type")
                plt.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)

                # Format axes
                ax = plt.gca()
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${int(x):,}"))
                ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y')) # Show only year
                ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator(base=max(1, len(df_plot_cleaned)//10))) # Auto-adjust year ticks
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout() # Adjust layout

                # Save to buffer and upload
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                plt.close() # Close plot to free memory

                filename = f"yearly_trend_single_region_{timestamp}.png"
                url = upload_visualization_to_s3(buffer.getvalue(), viz_folder, filename)
                if url:
                    # Find original column names corresponding to the ones used
                    original_date_col = original_columns[df_plot.columns.to_list().index(date_col)]
                    original_num_cols = [original_columns[df_plot.columns.to_list().index(c)] for c in valid_numeric_cols]
                    visualizations.append({
                        "url": url,
                        "type": "time_series_single_region",
                        "title": plot_title,
                        "columns": [original_date_col] + original_num_cols # Report original columns used
                    })
            else:
                plt.close()

        # === GRAPH 2: Cross-City Overall Average Grouped Bar Chart ===
        elif query_type == 'cross_city_comparison':
            # Identify city column and overall average columns
            city_col = 'CITY' # Assuming the column name is 'CITY' (standardized to upper)
            value_cols = sorted([c for c in df_plot.columns if c.startswith('OVERALL_AVG_')])

            if city_col not in df_plot.columns or not value_cols:
                return []

            # Select relevant columns and create a working copy
            df_clean = df_plot[[city_col] + value_cols].copy()

            # Convert value columns to numeric, coercing errors
            valid_value_cols = []
            for col in value_cols:
                if not pd.api.types.is_numeric_dtype(df_clean[col]):
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if not df_clean[col].isnull().all():
                     valid_value_cols.append(col)

            if not valid_value_cols:
                return []

            # Drop rows where city is missing or *all* valid value columns are NaN
            df_clean.dropna(subset=[city_col], inplace=True)
            df_clean.dropna(subset=valid_value_cols, how='all', inplace=True)

            if df_clean.empty:
                return []

            # Melt the dataframe for easy plotting with seaborn's grouped bar chart
            df_melted = pd.melt(df_clean,
                               id_vars=[city_col],
                               value_vars=valid_value_cols, # Use only valid cols
                               var_name='Metric',
                               value_name='Average Value')

            # Remove rows where the specific metric's average value is null (important after melt)
            df_melted.dropna(subset=['Average Value'], inplace=True)

            if df_melted.empty:
                return []

            # Clean up metric names for the legend
            df_melted['Metric'] = df_melted['Metric'].str.replace('OVERALL_AVG_', '').str.replace('_', ' ').str.replace('Sfr', 'SFR').str.title()

            # Determine plot size based on number of cities and metrics
            num_cities = df_clean[city_col].nunique()
            fig_width = max(8, num_cities * 1.5) # Adjust width based on cities
            fig_height = 6 + num_cities * 0.1   # Adjust height slightly

            # --- Plotting ---
            plt.figure(figsize=(fig_width, fig_height))
            plot_title = f"Overall Average Home Value Comparison {filters_str_city}"
            ylabel = "Overall Average Value ($)"
            xlabel = "City"

            # Create the grouped bar chart
            sns.barplot(data=df_melted, x=city_col, y='Average Value', hue='Metric', palette='viridis') # 'viridis' is a nice palette

            plt.xlabel(xlabel, fontsize=12)
            plt.ylabel(ylabel, fontsize=12)
            plt.title(plot_title, fontsize=14, fontweight='bold')
            plt.xticks(rotation=45, ha='right') # Rotate city names if needed

            # Format Y-axis as currency
            ax = plt.gca()
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${int(x):,}"))
            plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

            # Adjust legend position
            plt.legend(title='Home Type', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)

            plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make room for legend (may need tuning)

            # --- Save Plot ---
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight') # Use bbox_inches='tight'
            buffer.seek(0)
            plt.close()

            filename = f"overall_avg_cross_city_grouped_{timestamp}.png"
            url = upload_visualization_to_s3(buffer.getvalue(), viz_folder, filename)
            if url:
                # Find original column names corresponding to the ones used
                original_city_col = original_columns[df_plot.columns.to_list().index(city_col)]
                original_value_cols = [original_columns[df_plot.columns.to_list().index(c)] for c in valid_value_cols]
                visualizations.append({
                    "url": url,
                    "type": "grouped_bar_cross_city_avg",
                    "title": plot_title,
                    "columns": [original_city_col] + original_value_cols # Report original columns used
                })

        return visualizations

    except Exception as e:
        print(f"Error during visualization generation: {e}")
        plt.close() # Ensure plot is closed even if error occurs
        return [] # Return empty list indicating failure

# --- generate_data_summary (Handles single region trend and cross-city multi-metric overall avg) ---
def generate_data_summary(filters, single_region_df, cross_city_comp_df):
    """
    Generates a formatted markdown summary combining insights from both analyses.
    Includes the fix for the IndexError in the Synthesis section.

    Args:
        filters (dict): Dictionary containing filters like 'City', 'RegionName'.
        single_region_df (pd.DataFrame or None): DataFrame from the single region trend query.
        cross_city_comp_df (pd.DataFrame or None): DataFrame from the cross-city comparison query.

    Returns:
        str: A markdown formatted text summary.
    """
    print("\n--- Generating Combined Data Summary ---")

    # Check data availability
    has_single_data = single_region_df is not None and not single_region_df.empty
    has_cross_city_data = cross_city_comp_df is not None and not cross_city_comp_df.empty

    # Extract key filters for context
    specific_region = filters.get("RegionName", "N/A")
    city = filters.get("City", "N/A")

    summary_lines = []

    # === Overall Title ===
    summary_lines.append(f"## Real Estate Analysis: {city} (Region: {specific_region})")
    summary_lines.append(f"*(Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')})*") # Added time and UTC marker
    summary_lines.append("\n---\n")

    # === Section 1: Specific Region Yearly Analysis ===
    summary_lines.append(f"### Yearly Trends: Region {specific_region} in {city}")
    if has_single_data:
        # Work with a copy and standardize columns
        single_df = single_region_df.copy()
        single_df.columns = [str(col).upper() for col in single_df.columns]
        date_col = next((c for c in single_df.columns if 'DATE' in c), None)

        # Determine date range
        date_range_str = "an unknown period"
        if date_col and pd.api.types.is_datetime64_any_dtype(single_df[date_col]):
            valid_dates = single_df[date_col].dropna()
            if not valid_dates.empty:
                min_year = valid_dates.dt.year.min()
                max_year = valid_dates.dt.year.max()
                date_range_str = f"{min_year}" if min_year == max_year else f"{min_year} - {max_year}"
                date_range_str = f"between **{date_range_str}**" # Add formatting

        summary_lines.append(f"This section analyzes the average annual home value trends for region **{specific_region}** within **{city}**, {date_range_str}.")

        numeric_cols = sorted([c for c in single_df.columns if c.startswith('AVG_')])
        if numeric_cols:
            summary_lines.append("Key observations:")
            processed_metrics = 0
            for col in numeric_cols:
                col_name_pretty = col.replace('AVG_', '').replace('_', ' ').replace('Sfr', 'SFR').title()
                # Ensure data is numeric and not all NaN
                if col in single_df.columns and pd.api.types.is_numeric_dtype(single_df[col]) and not single_df[col].isnull().all():
                    # Use date_col if available for sorting and selecting first/last
                    if date_col and date_col in single_df.columns:
                         valid_numeric_df = single_df[[date_col, col]].dropna().sort_values(by=date_col)
                         valid_numeric_series = valid_numeric_df[col]
                    else:
                         valid_numeric_df = None # Indicate no date df available
                         valid_numeric_series = single_df[col].dropna()

                    if len(valid_numeric_series) > 1:
                        # Get first/last values, using df if available, else series
                        first_val = valid_numeric_df.iloc[0][col] if valid_numeric_df is not None else valid_numeric_series.iloc[0]
                        last_val = valid_numeric_df.iloc[-1][col] if valid_numeric_df is not None else valid_numeric_series.iloc[-1]
                        overall_avg_val = valid_numeric_series.mean()

                        # Calculate change
                        change_pct_str = "N/A"
                        direction = "changed"
                        if pd.notna(first_val) and pd.notna(last_val) and first_val != 0:
                            change_pct = ((last_val - first_val) / abs(first_val)) * 100 # Use absolute for percentage base
                            change_pct_str = f"{change_pct:+.1f}%"
                            if change_pct > 5: direction = "increased significantly"
                            elif change_pct >= 0.1: direction = "increased"
                            elif change_pct < -5: direction = "decreased significantly"
                            elif change_pct <= -0.1: direction = "decreased"
                            else: direction = "remained relatively stable"

                        # Format values
                        avg_val_str = f"${overall_avg_val:,.0f}" if pd.notna(overall_avg_val) else 'N/A'
                        first_val_str = f"${first_val:,.0f}" if pd.notna(first_val) else 'N/A'
                        last_val_str = f"${last_val:,.0f}" if pd.notna(last_val) else 'N/A'

                        summary_lines.append(f"*   **{col_name_pretty}:** The average value **{direction}** ({change_pct_str}) from {first_val_str} to {last_val_str} over the observed period, with an overall average around {avg_val_str}.")
                        processed_metrics += 1
                    elif len(valid_numeric_series) == 1:
                        avg_val_str = f"${valid_numeric_series.iloc[0]:,.0f}" if pd.notna(valid_numeric_series.iloc[0]) else 'N/A'
                        summary_lines.append(f"*   **{col_name_pretty}:** Only one data point available ({avg_val_str}). Trend analysis not possible.")
                        processed_metrics += 1
                    # else: implicitly handles cases with 0 valid points - no message needed
                # else: No valid numeric data for this column - implicitly skipped
            if processed_metrics == 0:
                 summary_lines.append("*   *No valid yearly average data found for any metric in this specific region after cleaning.*")
        else:
            summary_lines.append("*   *No yearly average columns (starting with 'AVG_') were found in the data for this region.*")
    else:
        summary_lines.append("*   *Data for yearly trends in this specific region was not available or the query failed.*")
    summary_lines.append("\n")

    # === Section 2: City Comparison Analysis (Multi-Metric) ===
    summary_lines.append(f"### City Comparison: Overall Average Home Values")
    if has_cross_city_data:
        # Work with a copy and standardize columns
        cross_city_df = cross_city_comp_df.copy()
        cross_city_df.columns = [str(col).upper() for col in cross_city_df.columns]
        city_col = 'CITY'
        value_cols = sorted([c for c in cross_city_df.columns if c.startswith('OVERALL_AVG_')])

        # Declare comp_df_clean early for use in Synthesis section
        comp_df_clean = pd.DataFrame()

        if city_col in cross_city_df.columns and value_cols:
            # Ensure value cols are numeric for calculations
            valid_value_cols = []
            for col in value_cols:
                 if col in cross_city_df.columns: # Check if column actually exists
                    if not pd.api.types.is_numeric_dtype(cross_city_df[col]):
                        cross_city_df[col] = pd.to_numeric(cross_city_df[col], errors='coerce')
                    if not cross_city_df[col].isnull().all():
                        valid_value_cols.append(col)

            # Keep only rows with a city name and at least one valid value
            comp_df_clean = cross_city_df.dropna(subset=[city_col])
            if valid_value_cols: # Only drop based on value cols if they exist
                comp_df_clean = comp_df_clean.dropna(subset=valid_value_cols, how='all').copy()
            else:
                comp_df_clean = comp_df_clean.copy() # Keep all rows if no valid value cols

            if not comp_df_clean.empty and valid_value_cols:
                num_cities = comp_df_clean[city_col].nunique()
                compared_cities = comp_df_clean[city_col].unique().tolist()
                summary_lines.append(f"This section compares the overall average home values across **{num_cities}** sample cities: {', '.join(compared_cities)} (based on available data).")

                # Report ranges for each metric
                summary_lines.append("\n*   **Value Ranges Across Cities:**")
                for col in valid_value_cols:
                    col_name_pretty = col.replace('OVERALL_AVG_', '').replace('_', ' ').replace('Sfr', 'SFR').title()
                    valid_data = comp_df_clean[col].dropna()
                    if not valid_data.empty:
                        min_val_str = f"${valid_data.min():,.0f}"
                        max_val_str = f"${valid_data.max():,.0f}"
                        summary_lines.append(f"    *   {col_name_pretty}: {min_val_str} - {max_val_str}")
                    # else: skip if no valid data for this metric

                # Highlight the user's selected city
                user_city_row = comp_df_clean[comp_df_clean[city_col] == city]
                if not user_city_row.empty:
                    summary_lines.append(f"\n*   **Details for {city}:**")
                    user_city_data = user_city_row.iloc[0]
                    has_user_data = False
                    # Use a key metric for ranking, e.g., SFR or the first available one
                    key_metric_rank = next((c for c in ['OVERALL_AVG_SFR', 'OVERALL_AVG_3BED', 'OVERALL_AVG_2BED', 'OVERALL_AVG_1BED'] if c in valid_value_cols), None)

                    for col in valid_value_cols:
                         col_name_pretty = col.replace('OVERALL_AVG_', '').replace('_', ' ').replace('Sfr', 'SFR').title()
                         user_val = user_city_data.get(col)
                         if pd.notna(user_val):
                             user_val_str = f"${user_val:,.0f}"
                             rank_str = ""
                             # Add ranking only for the chosen key metric
                             if key_metric_rank and col == key_metric_rank:
                                 try:
                                     # Sort by the key metric, handle NaNs by placing them last
                                     sorted_df = comp_df_clean[[city_col, key_metric_rank]].dropna(subset=[key_metric_rank]).sort_values(key_metric_rank, ascending=False)
                                     # Find rank (ensure city exists in the sorted list)
                                     city_rank_list = list(sorted_df[city_col])
                                     if city in city_rank_list:
                                         rank = city_rank_list.index(city) + 1
                                         num_ranked = len(sorted_df)
                                         position = ""
                                         if num_ranked == 1: position = "(only city with data)"
                                         elif rank == 1: position = "highest"
                                         elif rank == num_ranked: position = "lowest"
                                         else: position = f"rank {rank}/{num_ranked}"
                                         rank_str = f" (Overall {position} based on {col_name_pretty})"
                                 except Exception as rank_e:
                                      print(f"Debug: Error calculating rank for {key_metric_rank}: {rank_e}") # Debug rank issues
                                      rank_str = "" # Default to no rank string on error

                             summary_lines.append(f"    *   Avg {col_name_pretty}: **{user_val_str}**{rank_str}")
                             has_user_data = True

                    if not has_user_data:
                         summary_lines.append(f"    *   No specific average home value data found for {city} in this comparison dataset.")
                else:
                     summary_lines.append(f"\n*   *Data for **{city}** was not found among the {num_cities} cities compared (possibly due to missing values or not being in the sample).*")

            elif not valid_value_cols:
                 summary_lines.append("*   *No valid overall average columns found after cleaning. Cannot perform comparison.*")
            else: # comp_df_clean was empty
                 summary_lines.append("*   *No valid data found for any city in the comparison after cleaning.*")
        elif city_col not in cross_city_df.columns:
             summary_lines.append(f"*   *Required city column ('{city_col}') missing from the comparison data.*")
        else: # not value_cols
             summary_lines.append("*   *No overall average columns (starting with 'OVERALL_AVG_') were found in the comparison data.*")
    else:
        summary_lines.append("*   *Data for cross-city comparison was not available or the query failed.*")
    summary_lines.append("\n")


    # === Section 3: Overall Synthesis ===
    summary_lines.append("### Synthesis & Considerations")
    insight_parts = []
    if has_single_data:
        insight_parts.append(f"yearly trends within region **{specific_region}**")
    if has_cross_city_data:
        insight_parts.append(f"how **{city}** compares to other sample cities on overall average values")

    if len(insight_parts) == 2:
        # Try to get relative position from comparison data again for synthesis
        position_desc = "its market position"
        # Check if comp_df_clean was defined and is not empty
        if 'comp_df_clean' in locals() and isinstance(comp_df_clean, pd.DataFrame) and not comp_df_clean.empty:
             city_col_upper = 'CITY' # Already upper
             user_city_row = comp_df_clean[comp_df_clean[city_col_upper] == city.upper()] # Compare upper case if city filter might not match case
             if user_city_row.empty: # Try original case if upper failed
                  user_city_row = comp_df_clean[comp_df_clean[city_col_upper] == city]

             value_cols_upper = [c for c in comp_df_clean.columns if c.startswith('OVERALL_AVG_')]
             key_metric_rank = next((c for c in ['OVERALL_AVG_SFR', 'OVERALL_AVG_3BED', 'OVERALL_AVG_2BED', 'OVERALL_AVG_1BED'] if c in value_cols_upper), None)

             if not user_city_row.empty and key_metric_rank and pd.notna(user_city_row.iloc[0].get(key_metric_rank)):
                 try:
                     # Filter comp_df_clean for the key metric and drop NaNs before sorting
                     sorted_df = comp_df_clean[[city_col_upper, key_metric_rank]].dropna(subset=[key_metric_rank]).sort_values(key_metric_rank, ascending=False)
                     city_rank_list = list(sorted_df[city_col_upper])
                     # Find city in the list (case-insensitive check)
                     city_found_rank = next((c for c in city_rank_list if c.upper() == city.upper()), None)

                     if city_found_rank:
                         rank = city_rank_list.index(city_found_rank) + 1
                         num_ranked = len(sorted_df)
                         if num_ranked > 1:
                             if rank <= num_ranked / 3: position_desc = "among the higher-priced areas"
                             elif rank >= (num_ranked * 2 / 3): position_desc = "among the lower-priced areas"
                             else: position_desc = "in the mid-range price tier"
                             position_desc += f" (based on avg. {key_metric_rank.replace('OVERALL_AVG_', '').replace('_', ' ').title()})"
                         elif num_ranked == 1:
                              position_desc = "as the only comparable city with data for this metric"

                 except Exception as e_rank_synth:
                      print(f"Debug: Error calculating rank in Synthesis: {e_rank_synth}")
                      pass # Ignore errors deriving position

        summary_lines.append(f"Combining the analyses provides insights into both the {insight_parts[0]} and {insight_parts[1]}. The data suggests that **{city}** holds {position_desc} relative to the limited sample of cities compared.")
        summary_lines.append("Consider both levels of detail (regional trends and city-wide comparison) for a more complete picture. Note that the city comparison is based on a small sample.")

    elif has_single_data: # This block handles case where only single_region data exists
        summary_lines.append(f"The analysis focused on {insight_parts[0]}. Understanding how these trends compare to other areas or the city overall would require additional comparative data.")

    elif has_cross_city_data: # This block handles case where only cross_city data exists
        # --- CORRECTED LINE ---
        summary_lines.append(f"The analysis focused on {insight_parts[0]}. Examining specific neighborhood or regional trends within **{city}** would provide more granular insights.")
        # --- END CORRECTION ---

    else: # Neither data source was available
        summary_lines.append("Insufficient data was available from either the regional trend query or the city comparison query for a comprehensive analysis.")

    summary_lines.append("\n*Disclaimer: This analysis is based on the available data and the generated SQL queries. Data quality, completeness, and the limited scope of comparison may affect conclusions.*")

    return "\n".join(summary_lines)

# --- Direct SQL Query Templates ---
def get_fixed_query_templates(filter_dict):
    """
    Returns fixed SQL query templates with parameters filled in from filter_dict.
    No LLM calls, just direct string formatting for predictable queries.
    
    Args:
        filter_dict (dict): Dictionary containing 'City' and 'RegionName' for filtering
        
    Returns:
        tuple: (single_region_query, cross_city_query)
    """
    region = filter_dict.get("RegionName", "")
    city = filter_dict.get("City", "")
    
    if not region or not city:
        print("Error: Missing 'RegionName' or 'City' in filter_dict.")
        return None, None
    
    # Define table name and schema 
    db_schema_table = f"{SNOWFLAKE_DATABASE}.{SNOWFLAKE_SCHEMA}.MERGED_HOMEVALUES"
    
    # QUERY 1: Single Region Yearly Trend
    single_region_query = f"""
    SELECT 
        DATE_TRUNC('YEAR', "Date") AS "YEAR_START_DATE",
        AVG(TRY_TO_NUMBER("1Bed_HomeValue")) AS "AVG_1BED",
        AVG(TRY_TO_NUMBER("2Bed_HomeValue")) AS "AVG_2BED",
        AVG(TRY_TO_NUMBER("3Bed_HomeValue")) AS "AVG_3BED",
        AVG(TRY_TO_NUMBER("MA_Single_Family_HomeValues")) AS "AVG_SFR"
    FROM {db_schema_table}
    WHERE "RegionName" = '{region}' AND "City" = '{city}'
    GROUP BY "YEAR_START_DATE"
    ORDER BY "YEAR_START_DATE" ASC;
    """
    
    # QUERY 2: Cross-City Comparison with CTE
    cross_city_query = f"""
    WITH SampleCities AS (
        SELECT '{city}' AS "City"
        UNION 
        SELECT DISTINCT "City" FROM {db_schema_table}
        WHERE "City" != '{city}'
        LIMIT 4
    )
    SELECT 
        m."City",
        AVG(TRY_TO_NUMBER("1Bed_HomeValue")) AS "OVERALL_AVG_1BED",
        AVG(TRY_TO_NUMBER("2Bed_HomeValue")) AS "OVERALL_AVG_2BED",
        AVG(TRY_TO_NUMBER("3Bed_HomeValue")) AS "OVERALL_AVG_3BED",
        AVG(TRY_TO_NUMBER("MA_Single_Family_HomeValues")) AS "OVERALL_AVG_SFR"
    FROM {db_schema_table} AS m
    INNER JOIN SampleCities s ON m."City" = s."City"
    GROUP BY m."City"
    ORDER BY "OVERALL_AVG_SFR" DESC;
    """
    
    return single_region_query, cross_city_query

# --- Vision Analysis for Graph Interpretation ---
def analyze_visualization_with_gemini(visualization_urls, filter_dict):
    """Uses Gemini Pro Vision to analyze visualizations."""
    if not visualization_urls:
        return "No visualizations available for analysis."
        
    try:
        import requests
        
        # Get API key
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY not found in environment variables."
            
        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Extract context for prompt
        city = filter_dict.get("City", "Unknown City")
        region = filter_dict.get("RegionName", "Unknown Region")
        
        # Prepare prompt
        prompt = f"""
        Analyze these real estate market visualizations for {city} (region: {region}).
        Focus on:
        1. Key trends in home values
        2. Market position comparison
        3. Property type performance
        4. Notable patterns or anomalies
        
        Provide analysis in clear, concise bullet points.
        """
        
        # Process each visualization
        all_analyses = []
        for viz_info in visualization_urls:
            url = viz_info.get("url")
            if url and url.startswith("http"):
                try:
                    # Download image
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Generate content with image
                        response = genai.GenerativeModel(model_name="gemini-2.0-flash-exp").generate_content(
                            [
                                prompt,
                                {"mime_type": "image/png", "data": response.content}  # Use dictionary format
                            ]
                        )
                        all_analyses.append(response.text)
                except Exception as e:
                    print(f"Error processing visualization {url}: {e}")
                    continue
        
        if not all_analyses:
            return "Could not analyze any visualizations."
            
        # Combine all analyses
        combined_analysis = "\n\n".join([
            f"### Analysis {i+1}\n{analysis}" 
            for i, analysis in enumerate(all_analyses)
        ])
        
        return combined_analysis
        
    except Exception as e:
        print(f"Error in visualization analysis: {e}")
        return f"Error analyzing visualizations: {str(e)}"

# --- generate_snowflake_insights (Main function) ---
def generate_snowflake_insights(filter_dict):
    """
    Main orchestrator function:
    1. Uses fixed SQL templates instead of LLM-generated queries
    2. Executes both queries against Snowflake
    3. Generates visualizations for both results
    4. Generates a combined textual summary
    5. Optional visual analysis using Gemini Vision
    6. Packages the results
    """
    start_time = datetime.now()

    # Validate essential filters
    if not filter_dict.get("RegionName") or not filter_dict.get("City"):
        print("Error: Missing required filters (RegionName, City).")
        return {"summary": "Error: Missing required filters (RegionName, City).", "visualizations": [], "raw_data_sample": []}

    # 1. Get SQL queries from templates (no LLM involved)
    try:
        single_region_query, cross_city_query = get_fixed_query_templates(filter_dict)
        if not single_region_query or not cross_city_query:
            print("Error: Failed to generate SQL query templates.")
            return {"summary": "Error: Failed to generate SQL query templates.", "visualizations": [], "raw_data_sample": []}
    except Exception as e:
        print(f"Unexpected error during SQL template generation: {str(e)}")
        return {"summary": f"Error generating SQL templates: {str(e)}", "visualizations": [], "raw_data_sample": []}

    # Initialize results
    single_region_df = None
    cross_city_df = None
    visualizations = []
    summary = "Analysis could not be fully completed." # Default summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Unique timestamp for this run
    error_flag_single = False
    error_flag_cross = False
    error_messages = [] # Collect specific error notes

    # 2. Execute Query 1 (Single Region Trend)
    try:
        single_region_df = fetch_snowflake_df(single_region_query, "Single Region Trend Query")
        if single_region_df is None:
            error_messages.append("Execution failed for the specific region trend query.")
            error_flag_single = True
        elif single_region_df.empty:
            error_messages.append("No data found for the specific region trend query filters.") # Note no data
            # It's not an error, but summary needs to reflect it.
        else:
            # Generate visualization only if data is present
            vis1 = create_and_save_graph(single_region_df, 'single_region_trend', timestamp, filter_dict)
            visualizations.extend(vis1)
    except Exception as e:
        error_messages.append(f"Error executing region trend query: {str(e)}")
        error_flag_single = True
        print(f"Error executing region trend query: {str(e)}")

    # 3. Execute Query 2 (Cross-City Comparison)
    try:
        cross_city_df = fetch_snowflake_df(cross_city_query, "Cross-City Comparison Query")
        if cross_city_df is None:
            error_messages.append("Execution failed for the cross-city comparison query.")
            error_flag_cross = True
        elif cross_city_df.empty:
            error_messages.append("No data found for the cross-city comparison query filters.") # Note no data
        else:
            # Generate visualization only if data is present
            vis2 = create_and_save_graph(cross_city_df, 'cross_city_comparison', timestamp, filter_dict)
            visualizations.extend(vis2)
    except Exception as e:
        error_messages.append(f"Error executing cross-city comparison query: {str(e)}")
        error_flag_cross = True
        print(f"Error executing cross-city comparison query: {str(e)}")

    # 4. Generate data-based summary
    try:
        # Pass the DFs regardless of whether they are None, empty, or populated
        print(f"Single Region DataFrame: {single_region_df}")
        print(f"Cross City DataFrame: {cross_city_df}")
        numeric_summary = generate_data_summary(filter_dict, single_region_df, cross_city_df)
    except Exception as e:
        print(f"Error generating numeric summary: {str(e)}")
        numeric_summary = f"Error generating numeric summary: {str(e)}"
        if not error_messages:  # Only add this if no other errors
            error_messages.append(f"Error generating numeric summary: {str(e)}")
    
    # 5. Generate visual analysis if enabled and we have visualizations
    vision_analysis = None
    use_vision_analysis = filter_dict.get("use_vision_analysis", False)
    
    if use_vision_analysis and visualizations:
        try:
            print("Analyzing visualizations with Gemini Vision...")
            vision_analysis = analyze_visualization_with_gemini(visualizations, filter_dict)
        except Exception as e:
            print(f"Error during visualization analysis: {e}")
            vision_analysis = f"Error analyzing visualizations: {str(e)}"
    
    # Combine summaries
    if vision_analysis:
        summary = f"{numeric_summary}\n\n## Visualization Analysis\n\n{vision_analysis}"
    else:
        summary = numeric_summary
    
    # Prepend error/status notes to summary if any issues occurred or no data found
    if error_messages:
         error_prefix = "### Important Notes on Data Availability:\n" + "\n".join([f"- {msg}" for msg in error_messages]) + "\n\n---\n\n"
         summary = error_prefix + summary

    # 7. Final Result Packaging
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"Completed in {duration.total_seconds():.2f} seconds")

    # Determine status based on DataFrame content (None=Fail, Empty=NoData, Populated=Success)
    status_region = "Failed" if error_flag_single else "No Data" if single_region_df is not None and single_region_df.empty else "Success" if single_region_df is not None else "Skipped/Invalid"
    status_comp = "Failed" if error_flag_cross else "No Data" if cross_city_df is not None and cross_city_df.empty else "Success" if cross_city_df is not None else "Skipped/Invalid"

    # Extract raw data samples
    raw_data_sample = extract_raw_data_sample(single_region_df, cross_city_df)

    result = {
        "summary": summary,
        "visualizations": visualizations,
        "raw_data_sample": raw_data_sample,
        "metadata": {
            "filters_used": filter_dict,
            "timestamp_utc": start_time.isoformat() + "Z", # ISO format UTC
            "duration_seconds": round(duration.total_seconds(), 2),
            "region_query_status": status_region,
            "comparison_query_status": status_comp,
            "error_messages": error_messages if error_messages else None,
            "vision_analysis_used": vision_analysis is not None
        }
    }
    return result

def extract_raw_data_sample(single_region_df, cross_city_df, max_rows=5):
    """Extract sample data from both DataFrames for debugging/display."""
    sample_data = {
        "single_region_trend": None,
        "cross_city_comparison": None
    }
    
    if single_region_df is not None and not single_region_df.empty:
        sample_data["single_region_trend"] = single_region_df.head(max_rows).to_dict('records')
        print(f"Single Region Trend Sample Data: {sample_data['single_region_trend']}")
    if cross_city_df is not None and not cross_city_df.empty:
        sample_data["cross_city_comparison"] = cross_city_df.head(max_rows).to_dict('records')
        print(f"Cross City Comparison Sample Data: {sample_data['cross_city_comparison']}")
        
    return sample_data

# # =============================================
# # Main execution block
# # =============================================
# if __name__ == "__main__":
#     print("===== Real Estate Analysis =====")

#     # --- Define Filters ---
#     filter_dict = {
#         "RegionName": "2129",  # Zip code in Boston
#         "City": "Boston",      # City
#     }
#     print(f"Filter: {filter_dict}")

#     # --- Run Analysis ---
#     insights = generate_snowflake_insights(filter_dict)

#     # --- Display Results ---
#     print("\n=== ANALYSIS RESULTS ===")
#     print(f"- Visualizations: {len(insights.get('visualizations', []))}")
#     if insights.get('visualizations'):
#         for i, viz in enumerate(insights['visualizations']):
#             print(f"  {i+1}. {viz.get('type')}: {viz.get('url')}")
    
#     print(f"- Data Sample Rows: {len(insights.get('raw_data_sample', []))}")
#     print(f"- Execution Time: {insights.get('metadata', {}).get('duration_seconds', 0):.2f} seconds")
#     print("\n=== End of Analysis ===")
#     print("Insights: ", insights)
