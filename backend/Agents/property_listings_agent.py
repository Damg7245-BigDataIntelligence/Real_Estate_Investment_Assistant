from typing import Dict, Any, List, Tuple
import snowflake.connector
import os
from dotenv import load_dotenv
import sys

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from backend.agents.gemini_visualization_agent import GeminiVisualizationAgent

class PropertyListingsAgent:
    def __init__(self):
        load_dotenv()
        self.conn_config = {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "role": os.getenv("SNOWFLAKE_ROLE"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": "CURRENT_LISTINGS"
        }
        self.gemini_agent = GeminiVisualizationAgent()

    def get_property_listings(self, zip_codes: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Fetch property listings from Snowflake for given zip codes
        
        Args:
            zip_codes (List[str]): List of ZIP codes to fetch properties for
            
        Returns:
            Tuple[List[Dict[str, Any]], Dict[str, int]]: 
                - List of property listings with all details
                - Dictionary with count of listings per zip code
        """
        try:
            print("Getting property listings for zip codes")
            # Remove .0 from zip codes for listings query
            clean_zip_codes = [zip_code.replace('.0', '') for zip_code in zip_codes]
            
            # Format zip codes for SQL IN clause
            zip_codes_str = ", ".join(f"'{zip_code}'" for zip_code in clean_zip_codes)
            print("zip_codes_str", zip_codes_str)
            
            # First, get count of listings per zip code
            count_query = f"""
                SELECT 
                    "RegionName" as zip_code,
                    COUNT(*) as listing_count
                FROM LISTINGS
                WHERE "RegionName" IN ({zip_codes_str})
                GROUP BY "RegionName";
            """ 
            # SQL query to fetch all properties for given zip codes
            listings_query = f"""
                SELECT 
                    "City",
                    "Address",
                    "StateName",
                    "RegionName" as "ZipCode",
                    "Price",
                    "Beds",
                    "Baths",
                    "Sqft",
                    "Url",
                    "Date"
                FROM LISTINGS
                WHERE "RegionName" IN ({zip_codes_str})
                ORDER BY "Price" ASC
                LIMIT 10;
            """
            
            # Connect to Snowflake and execute queries
            with snowflake.connector.connect(**self.conn_config) as conn:
                with conn.cursor() as cursor:
                    # Get counts first
                    cursor.execute(count_query)
                    count_results = cursor.fetchall()
                    zip_code_counts = {row[0]: row[1] for row in count_results}
                    
                    # Check which zip codes have no listings
                    zip_codes_no_listings = [
                        zip_code for zip_code in zip_codes 
                        if zip_code not in zip_code_counts
                    ]
                    
                    # Get listings if any exist
                    listings = []
                    if zip_code_counts:
                        cursor.execute(listings_query)
                        columns = [desc[0] for desc in cursor.description]
                        results = cursor.fetchall()
                        listings = [dict(zip(columns, row)) for row in results]
                    
                    # Print summary
                    print("\nListing Summary:")
                    print("-" * 80)
                    for zip_code in clean_zip_codes:
                        count = zip_code_counts.get(zip_code, 0)
                        status = f"{count} listings found" if count > 0 else "No listings available"
                        print(f"ZIP Code {zip_code}: {status}")
                    print("-" * 80)
                    
                    print(f"\nFound total of {len(listings)} properties")
                    
                    return listings, zip_code_counts
                    
        except Exception as e:
            print(f"Error fetching property listings: {str(e)}")
            return [], {}

    def get_property_listings_with_analysis(self, zip_codes: List[str]) -> Dict[str, Any]:
        """
        Fetch property listings and generate analysis with visualization
        
        Args:
            zip_codes (List[str]): List of ZIP codes to fetch properties for
            
        Returns:
            Dict[str, Any]: Dictionary containing listings, summary, and visualization URLs
        """
        try:
            # Get the listings first
            listings, zip_counts = self.get_property_listings(zip_codes)
            print("listings", listings)
            print("zip_counts", zip_counts)
            if not listings:
                return {
                    "success": False,
                    "message": "No listings found for the provided ZIP codes",
                    "listings": [],
                    "summary": None,
                    "visualization": None
                }
            
            # Generate visualization and description
            visualization_result = self.gemini_agent.generate_property_visualization(listings)
            
            # Generate detailed summary
            summary = self.gemini_agent.generate_detailed_summary(listings)
            
            return {
                "success": True,
                "message": "Analysis completed successfully",
                "listings": listings,
                "summary": summary,
                "visualization": visualization_result,  # Now includes both URLs and description
                "zip_counts": zip_counts
            }
            
        except Exception as e:
            print(f"Error in property analysis: {str(e)}")
            return {
                "success": False,
                "message": f"Error during analysis: {str(e)}",
                "listings": [],
                "summary": None,
                "visualization": None
            }