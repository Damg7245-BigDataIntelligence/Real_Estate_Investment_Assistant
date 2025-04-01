import logging
from typing import Dict, Any, List
from dotenv import load_dotenv
import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from backend.Pipeline.pinecone_manager import RealEstateVectorDB

load_dotenv()

class ReviewsAgent:
    def __init__(self):
        self.pinecone_manager = RealEstateVectorDB()
        self.logger = logging.getLogger(__name__)
        
    async def search_area_insights(self, zipcode: str, category: str = None) -> Dict[str, Any]:
        """
        Search for comprehensive area insights
        
        Args:
            zipcode (str): ZIP code to search for
            category (str, optional): Category to focus on (safety, schools, amenities)
            
        Returns:
            Dict[str, Any]: Area insights and metadata
        """
        try:
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
            
            return {
                "success": True,
                "insights": area_insights,
                "zipcode": zipcode,
                "category": category
            }
            
        except Exception as e:
            self.logger.error(f"Error searching area insights: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def search_property_reviews(self, address: str) -> Dict[str, Any]:
        """
        Search for comprehensive property reviews and details
        
        Args:
            address (str): Property address to search for
            
        Returns:
            Dict[str, Any]: Property reviews and metadata
        """
        try:
            query = f"What are the reviews and details for {address}?"
            results = self.pinecone_manager.search_properties(query)
            
            # Filter for property information with improved context
            property_info = [r for r in results if (
                r['metadata']['type'] in ['property_summary', 'property_review'] and
                r['metadata'].get('address') == address
            )]
            
            return {
                "success": True,
                "reviews": property_info,
                "address": address
            }
            
        except Exception as e:
            self.logger.error(f"Error searching property reviews: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def test_search_functionality(self):
        """Test the search functionality with default values"""
        try:
            # Test area insights with default values
            print("\nTesting area insights search...")
            test_zipcode = "02127"
            categories = ['safety', 'schools', 'amenities']
            
            for category in categories:
                print(f"\nSearching {category} insights for ZIP code {test_zipcode}")
                results = await self.search_area_insights(test_zipcode, category)
                if results["success"]:
                    print(f"Found {len(results['insights'])} {category} insights")
                    for insight in results['insights'][:2]:  # Show first 2 insights
                        print(f"\nScore: {insight['score']}")
                        print(f"Text: {insight['metadata']['text'][:200]}...")
                else:
                    print(f"Error: {results['error']}")
            
            # Test property reviews with default address
            print("\nTesting property reviews search...")
            test_address = "9 M St #5, Boston, MA 02127"
            results = await self.search_property_reviews(test_address)
            
            if results["success"]:
                print(f"Found {len(results['reviews'])} reviews")
                for review in results['reviews'][:2]:  # Show first 2 reviews
                    print(f"\nScore: {review['score']}")
                    print(f"Type: {review['metadata']['type']}")
                    print(f"Text: {review['metadata']['text'][:200]}...")
            else:
                print(f"Error: {results['error']}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error in test functionality: {e}")
            return False

if __name__ == "__main__":
    import asyncio
    
    async def test():
        agent = ReviewsAgent()
        success = await agent.test_search_functionality()
        print(f"\nTest {'succeeded' if success else 'failed'}")
    
    asyncio.run(test())