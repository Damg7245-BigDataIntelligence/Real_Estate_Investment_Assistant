import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
from pinecone_manager import RealEstateVectorDB
import pandas as pd
import asyncio

load_dotenv()

class ReviewsAgent:
    def __init__(self):
        self.pinecone_manager = RealEstateVectorDB()
        self.logger = logging.getLogger(__name__)

    async def process_listings(self, csv_path: str):
        """Process listings and store in Pinecone with comprehensive data"""
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            listings = df.to_dict('records')
            
            for listing in listings:
                # Get comprehensive area and property data
                area_data = await self.pinecone_manager._get_complete_area_data(
                    str(listing['RegionName'])
                )
                property_data = await self.pinecone_manager._get_complete_property_data(
                    listing['Address']
                )
                
                # Store vectors with proper chunking
                await self.pinecone_manager._store_vectors(listing, area_data, property_data)
                
            self.logger.info(f"Processed {len(listings)} listings")
            
        except Exception as e:
            self.logger.error(f"Error processing listings: {e}")
            raise

    async def search_area_insights(self, zipcode: str, category: str = None) -> Dict[str, Any]:
        """Search for comprehensive area insights"""
        query = f"What are the key insights about {zipcode}?"
        if category:
            query += f" Focus on {category}."
            
        results = self.pinecone_manager.search_properties(query, zipcode)
        
        # Filter for area insights with improved context
        area_insights = [r for r in results if (
            r['metadata']['type'] in ['area_summary', 'area'] and
            (not category or  # If category specified, check context
             (category == 'safety' and r['metadata'].get('chunk_context', {}).get('mentions_safety')) or
             (category == 'schools' and r['metadata'].get('chunk_context', {}).get('mentions_schools')) or
             (category == 'amenities' and r['metadata'].get('chunk_context', {}).get('mentions_amenities'))
            )
        )]
        
        return area_insights

    async def search_property_reviews(self, address: str) -> Dict[str, Any]:
        """Search for comprehensive property reviews and details"""
        query = f"What are the reviews and details for {address}?"
        results = self.pinecone_manager.search_properties(query)
        
        # Filter for property information with improved context
        property_info = [r for r in results if (
            r['metadata']['type'] in ['property_summary', 'property_review'] and
            r['metadata'].get('address') == address
        )]
        
        return property_info

async def main():
    try:
        # Initialize agent
        agent = ReviewsAgent()
        
        # Process and store data with comprehensive information
        await agent.process_listings("datasets/Property_listings_data_redfin.csv")
        
        # Read CSV file for testing
        df = pd.read_csv("datasets/Property_listings_data_redfin.csv")
        listings = df.to_dict('records')
        
        if listings:
            # Get first zipcode and address from the data for testing
            sample_zipcode = str(listings[0]['RegionName'])
            sample_address = listings[0]['Address']
            
            # Search for insights with different categories
            print(f"\nSearching insights for zipcode {sample_zipcode}:")
            
            # Test different category searches
            categories = ['safety', 'schools', 'amenities']
            for category in categories:
                print(f"\nCategory: {category}")
                area_insights = await agent.search_area_insights(sample_zipcode, category=category)
                print(f"Found {len(area_insights)} insights:")
                for insight in area_insights:
                    print(f"\nScore: {insight['score']}")
                    print(f"Text: {insight['metadata']['text'][:200]}...")
            
            # Search for property details
            print(f"\nSearching reviews for address {sample_address}:")
            property_reviews = await agent.search_property_reviews(sample_address)
            print(f"Found {len(property_reviews)} reviews/details:")
            for review in property_reviews:
                print(f"\nScore: {review['score']}")
                print(f"Type: {review['metadata']['type']}")
                print(f"Text: {review['metadata']['text'][:200]}...")
        
    except Exception as e:
        print(f"Error processing data: {e}")

if __name__ == "__main__":
    asyncio.run(main())