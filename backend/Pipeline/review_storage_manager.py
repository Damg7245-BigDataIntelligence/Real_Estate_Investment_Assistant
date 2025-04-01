import logging
from typing import Dict, Any
from dotenv import load_dotenv
from backend.Pipeline.pinecone_manager import RealEstateVectorDB
import pandas as pd

load_dotenv()

class ReviewStorageManager:
    def __init__(self):
        self.pinecone_manager = RealEstateVectorDB()
        self.logger = logging.getLogger(__name__)

    async def process_and_store_listings(self, csv_path: str):
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
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing listings: {e}")
            return False

    async def test_storage_pipeline(self):
        """Test the storage pipeline with sample data"""
        try:
            # Test with a small sample dataset
            sample_data = {
                'RegionName': ['02108', '02109'],
                'Address': ['123 Main St, Boston, MA', '456 Park Ave, Boston, MA'],
                'Price': [500000, 600000],
                'Description': ['Sample property 1', 'Sample property 2']
            }
            
            df = pd.DataFrame(sample_data)
            temp_csv = "temp_sample.csv"
            df.to_csv(temp_csv, index=False)
            
            success = await self.process_and_store_listings(temp_csv)
            import os
            os.remove(temp_csv)  # Clean up
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in test pipeline: {e}")
            return False

if __name__ == "__main__":
    import asyncio
    
    async def test():
        manager = ReviewStorageManager()
        success = await manager.test_storage_pipeline()
        print(f"Test pipeline {'succeeded' if success else 'failed'}")
    
    asyncio.run(test())