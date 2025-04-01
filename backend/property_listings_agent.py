from typing import Dict, Any, List, Tuple
import snowflake.connector
import os
from dotenv import load_dotenv
from gemini_visualization_agent import GeminiVisualizationAgent

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
            # Format zip codes for SQL IN clause
            zip_codes_str = ", ".join(f"'{zip_code}'" for zip_code in zip_codes)
            
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
                ORDER BY "Price" ASC;
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
                    for zip_code in zip_codes:
                        count = zip_code_counts.get(zip_code, 0)
                        status = f"{count} listings found" if count > 0 else "No listings available"
                        print(f"ZIP Code {zip_code}: {status}")
                    print("-" * 80)
                    
                    if zip_codes_no_listings:
                        print("\nWarning: No listings found for the following ZIP codes:")
                        print(", ".join(zip_codes_no_listings))
                    
                    if listings:
                        print(f"\nFound total of {len(listings)} properties")
                        for listing in listings:
                            print("\nProperty Details:")
                            print(f"Address: {listing['Address']}, {listing['City']}, {listing['StateName']} {listing['ZipCode']}")
                            print(f"Price: ${listing['Price']}")
                            print(f"Specs: {listing['Beds']} beds, {listing['Baths']} baths, {listing['Sqft']} sqft")
                            print(f"Listing URL: {listing['Url']}")
                            print(f"Listed Date: {listing['Date']}")
                            print("-" * 80)
                    else:
                        print("\nNo listings found for any of the provided ZIP codes.")
                    
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
            Dict[str, Any]: Dictionary containing listings, summary, and visualization URL
        """
        try:
            # Get the listings first
            listings, zip_counts = self.get_property_listings(zip_codes)
            
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
                "visualization": visualization_result,  # Now includes both URL and description
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

if __name__ == "__main__":
    agent = PropertyListingsAgent()
    # Test with mix of valid and invalid zip codes
    zip_codes = ["2118", "2128", "2134"]  # 99999 is an invalid zip code
    result = agent.get_property_listings_with_analysis(zip_codes)

    if result["success"]:
        print("\nVisualization URL:", result["visualization"]["url"])
        print("\nDetailed Analysis:")
        print(result["summary"])
        print("\nListings found:", len(result["listings"]))
        print("ZIP code distribution:", result["zip_counts"])
    else:
        print("Error:", result["message"])
