import os
import logging
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from serpapi.google_search import GoogleSearch
from datetime import datetime
from typing import List, Dict
import textwrap
import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from llm_service import LLMService
import json
import re

class RealEstateVectorDB:
    def __init__(self):
        # Configure Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        load_dotenv(override=True)
        # Load environment variables
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        print(self.PINECONE_API_KEY)
        self.SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index_name = "real-estate-reviews"
        self.dimension = 384
        
        # Initialize embedding model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # List existing indexes before initialization
        self.list_existing_indexes()
        
        # Initialize index
        self._init_index()
        
        self.llm_service = LLMService()
        self.chunk_size = 512  # Optimal size for text chunks
        
    def list_existing_indexes(self):
        """List all existing Pinecone indexes in the project"""
        try:
            indexes = self.pc.list_indexes()
            logging.info(f"\nExisting Pinecone indexes ({len(indexes)}):")
            for idx in indexes:
                logging.info(f"- Name: {idx['name']}")
                logging.info(f"  Status: {idx.get('status', 'unknown')}")
                logging.info(f"  Host: {idx.get('host', 'unknown')}")
                logging.info(f"  Dimension: {idx.get('dimension', 'unknown')}")
                logging.info(f"  Metric: {idx.get('metric', 'unknown')}\n")
            return indexes
        except Exception as e:
            logging.error(f"Error listing Pinecone indexes: {e}")
            return []

    def delete_index(self, index_name: str):
        """Delete a specific Pinecone index"""
        try:
            if index_name in [idx["name"] for idx in self.pc.list_indexes()]:
                self.pc.delete_index(index_name)
                logging.info(f"Successfully deleted index: {index_name}")
            else:
                logging.warning(f"Index {index_name} not found")
        except Exception as e:
            logging.error(f"Error deleting index {index_name}: {e}")

    def _init_index(self):
        """Initialize or get Pinecone index"""
        try:
            # Check if index exists
            if self.index_name not in [index["name"] for index in self.pc.list_indexes()]:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logging.info(f"Created new index: {self.index_name}")
            
            self.index = self.pc.Index(self.index_name)
            logging.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logging.error(f"Error initializing Pinecone index: {e}")
            raise

    def process_listings_data(self, listings):
        """Process listings and store in Pinecone"""
        try:
            total = len(listings)
            logging.info(f"Starting to process {total} listings...")
            
            for idx, listing in enumerate(listings, 1):
                try:
                    # Ensure RegionName and Address exist
                    if 'RegionName' not in listing or 'Address' not in listing:
                        logging.warning(f"Skipping listing {idx}: Missing required fields")
                        continue
                        
                    logging.info(f"Processing listing {idx}/{total}: {listing['Address']}")
                    
                    # Get area data for the zip code
                    area_data = self._get_area_data(str(listing['RegionName']))
                    print(area_data)
                    # Get property-specific data
                    property_data = self._get_property_data(listing['Address'])
                    print(property_data)

                    # Store vectors in Pinecone
                    self._store_vectors(listing, area_data, property_data)
                    
                    logging.info(f"Successfully processed listing {idx}/{total}")
                    
                except Exception as e:
                    logging.error(f"Error processing listing {idx}: {str(e)}")
                    continue
                    
        except Exception as e:
            logging.error(f"Error in process_listings_data: {str(e)}")
            raise

    def load_property_data(self, csv_path: str):
        """Load and process property data from CSV"""
        try:
            # Verify the CSV path exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found at: {csv_path}")
                
            # Read CSV file with proper error handling
            logging.info(f"Attempting to read CSV from: {csv_path}")
            df = pd.read_csv(csv_path)
            
            # Convert ZIP codes to strings to avoid integer formatting issues
            if 'RegionName' in df.columns:
                df['RegionName'] = df['RegionName'].astype(str).str.zfill(5)
            
            # Convert to records
            listings = df.to_dict('records')
            logging.info(f"Loaded {len(listings)} properties from CSV")
            
            if listings:
                logging.info(f"Sample listing format:")
                logging.info(f"Columns available: {df.columns.tolist()}")
                logging.info(f"First listing: {listings[0]}")
                
                # Process the listings
                self.process_listings_data(listings)
            else:
                logging.warning("No listings found in the CSV file")
                
        except pd.errors.EmptyDataError:
            logging.error("The CSV file is empty")
            raise
        except Exception as e:
            logging.error(f"Error loading property data: {str(e)}")
            raise

    def _get_area_data(self, zipcode: str) -> list:
        """Get structured area data using targeted searches"""
        try:
            area_data = []
            
            # Search for schools
            school_search = GoogleSearch({
                "engine": "google",
                "q": f"best public and private schools ratings reviews {zipcode} Boston MA",
                "location": "Boston, Massachusetts",
                "api_key": self.SERPAPI_KEY,
                "num": 5,
                "type": "search"
            })
            school_results = school_search.get_dict().get("organic_results", [])
            for result in school_results:
                area_data.append({
                    "type": "school",
                    "title": result.get("title", ""),
                    "content": result.get("snippet", ""),
                    "rating": result.get("rating", None),
                    "source": "school_search"
                })

            # Search for neighborhood safety and crime
            safety_search = GoogleSearch({
                "engine": "google",
                "q": f"neighborhood safety crime statistics {zipcode} Boston MA",
                "location": "Boston, Massachusetts",
                "api_key": self.SERPAPI_KEY,
                "num": 5
            })
            safety_results = safety_search.get_dict().get("organic_results", [])
            for result in safety_results:
                area_data.append({
                    "type": "safety",
                    "title": result.get("title", ""),
                    "content": result.get("snippet", ""),
                    "source": "safety_search"
                })

            # Search for neighborhood amenities and lifestyle
            amenities_search = GoogleSearch({
                "engine": "google",
                "q": f"neighborhood amenities restaurants parks shopping {zipcode} Boston MA reviews",
                "location": "Boston, Massachusetts",
                "api_key": self.SERPAPI_KEY,
                "num": 5
            })
            amenities_results = amenities_search.get_dict().get("organic_results", [])
            for result in amenities_results:
                area_data.append({
                    "type": "amenities",
                    "title": result.get("title", ""),
                    "content": result.get("snippet", ""),
                    "source": "amenities_search"
                })

            return area_data

        except Exception as e:
            logging.error(f"Error getting area data for {zipcode}: {e}")
            return []

    def _get_property_data(self, address: str) -> list:
        """Get structured property data using targeted searches"""
        try:
            # Search Google Maps for property reviews
            maps_search = GoogleSearch({
                "engine": "google_maps",
                "q": address,
                "type": "reviews",
                "api_key": self.SERPAPI_KEY
            })
            maps_results = maps_search.get_dict()
            
            property_data = []
            
            # Process reviews if available
            reviews = maps_results.get("reviews", [])
            for review in reviews:
                property_data.append({
                    "type": "property_review",
                    "rating": review.get("rating"),
                    "content": review.get("snippet", ""),
                    "date": review.get("date", ""),
                    "source": "google_maps"
                })

            # Get property details
            if "place_results" in maps_results:
                place = maps_results["place_results"]
                property_data.append({
                    "type": "property_details",
                    "title": place.get("title", ""),
                    "rating": place.get("rating"),
                    "description": place.get("description", ""),
                    "address": place.get("address", ""),
                    "source": "google_maps"
                })

            return property_data

        except Exception as e:
            logging.error(f"Error getting property data for {address}: {e}")
            return []

    def _store_vectors(self, property_row: dict, area_data: list, property_data: list):
        """Store vectors with improved metadata and deduplication"""
        try:
            vectors = []
            seen_content = set()  # For deduplication
            
            # Process area data
            for item in area_data:
                # Create unique content key for deduplication
                content_key = f"{item['type']}:{item['content']}"
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                
                # Create enriched text for better semantic search
                text = f"""
                Type: {item['type']}
                Title: {item.get('title', '')}
                Content: {item['content']}
                Additional Info: {item.get('rating', 'No rating')} from {item['source']}
                """
                
                vector = {
                    "id": f"{item['type']}_{property_row['RegionName']}_{len(vectors)}_{datetime.now().timestamp()}",
                    "values": self.model.encode(text).tolist(),
                    "metadata": {
                        "type": item['type'],
                        "text": text,
                        "zipcode": property_row['RegionName'],
                        "source": item['source'],
                        "rating": item.get('rating'),
                        "title": item.get('title', '')
                    }
                }
                vectors.append(vector)
            
            # Process property data
            for item in property_data:
                content_key = f"{item['type']}:{item.get('content', '')}"
                if content_key in seen_content:
                    continue
                seen_content.add(content_key)
                
                text = f"""
                Type: {item['type']}
                Address: {property_row['Address']}
                Rating: {item.get('rating', 'No rating')}
                Content: {item.get('content', '')}
                Details: {item.get('description', '')}
                """
                
                vector = {
                    "id": f"{item['type']}_{property_row['Address']}_{len(vectors)}_{datetime.now().timestamp()}",
                    "values": self.model.encode(text).tolist(),
                    "metadata": {
                        "type": item['type'],
                        "text": text,
                        "address": property_row['Address'],
                        "zipcode": property_row['RegionName'],
                        "rating": item.get('rating'),
                        "source": item['source']
                    }
                }
                vectors.append(vector)
            
            # Batch upsert to Pinecone
            if vectors:
                self.index.upsert(vectors=vectors)
                logging.info(f"Stored {len(vectors)} unique vectors for {property_row['Address']}")
                
        except Exception as e:
            logging.error(f"Error storing vectors: {e}")

    def _store_vectors(self, property_row: pd.Series, area_data: list, property_data: list):
        """Store property and area data in Pinecone"""
        try:
            vectors = []
            
            # Create property vector
            property_text = f"""
            Property Details:
            Address: {property_row['Address']}
            Location: {property_row['RegionName']} (ZIP Code)
            """
            property_embedding = self.model.encode(property_text).tolist()
            
            vectors.append({
                "id": f"prop_{property_row['Address']}_{datetime.now().timestamp()}",
                "values": property_embedding,
                "metadata": {
                    "text": property_text,
                    "type": "property",
                    "address": property_row['Address'],
                    "zipcode": property_row['RegionName']
                }
            })
            
            # Create area vectors
            for i, result in enumerate(area_data):
                area_text = f"{result.get('title', '')}\n{result.get('snippet', '')}"
                area_embedding = self.model.encode(area_text).tolist()
                
                vectors.append({
                    "id": f"area_{property_row['RegionName']}_{i}_{datetime.now().timestamp()}",
                    "values": area_embedding,
                    "metadata": {
                        "text": area_text,
                        "type": "area",
                        "zipcode": property_row['RegionName']
                    }
                })
            
            # Create review vectors
            for i, review in enumerate(property_data):
                review_text = review.get('text', '')
                if review_text:
                    review_embedding = self.model.encode(review_text).tolist()
                    
                    vectors.append({
                        "id": f"review_{property_row['Address']}_{i}_{datetime.now().timestamp()}",
                        "values": review_embedding,
                        "metadata": {
                            "text": review_text,
                            "type": "review",
                            "address": property_row['Address'],
                            "zipcode": property_row['RegionName']
                        }
                    })
            
            # Batch upsert to Pinecone
            self.index.upsert(vectors=vectors)
            logging.info(f"Stored {len(vectors)} vectors for {property_row['Address']}")
            
        except Exception as e:
            logging.error(f"Error storing vectors: {e}")

    def search_properties(self, query: str, zipcode: str = None) -> list:
        """Search for properties and insights"""
        try:
            # Create query embedding
            query_embedding = self.model.encode(query).tolist()
            
            # Prepare filter
            filter_dict = {}
            if zipcode:
                filter_dict["zipcode"] = {"$eq": zipcode}
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                filter=filter_dict,
                top_k=5,
                include_metadata=True
            )
            
            return results.get("matches", [])
            
        except Exception as e:
            logging.error(f"Error searching properties: {e}")
            return []

    async def _get_complete_area_data(self, zipcode: str) -> dict:
        """Get comprehensive area data using multiple targeted searches"""
        try:
            area_data = {
                "schools": [],
                "safety": [],
                "amenities": [],
                "demographics": [],
                "market_trends": []
            }
            
            # School information
            school_search = GoogleSearch({
                "engine": "google",
                "q": f"schools ratings reviews {zipcode} Boston MA education quality",
                "location": "Boston, Massachusetts",
                "api_key": self.SERPAPI_KEY,
                "num": 10,
                "gl": "us"
            })
            area_data["schools"] = school_search.get_dict().get("organic_results", [])
            
            # Safety and crime statistics
            safety_search = GoogleSearch({
                "engine": "google",
                "q": f"crime statistics safety rating {zipcode} Boston MA neighborhood security",
                "location": "Boston, Massachusetts",
                "api_key": self.SERPAPI_KEY,
                "num": 10
            })
            area_data["safety"] = safety_search.get_dict().get("organic_results", [])
            
            # Amenities and lifestyle
            amenities_search = GoogleSearch({
                "engine": "google",
                "q": f"neighborhood amenities restaurants parks shopping {zipcode} Boston MA lifestyle",
                "location": "Boston, Massachusetts",
                "api_key": self.SERPAPI_KEY,
                "num": 10
            })
            area_data["amenities"] = amenities_search.get_dict().get("organic_results", [])
            
            # Get LLM summary of area data
            summary_prompt = f"""
            Summarize the following neighborhood information for {zipcode} Boston MA:

            Schools:
            {self._format_search_results(area_data['schools'])}

            Safety:
            {self._format_search_results(area_data['safety'])}

            Amenities:
            {self._format_search_results(area_data['amenities'])}

            Provide a comprehensive summary that covers:
            1. School quality and educational opportunities
            2. Crime rates and safety metrics
            3. Local amenities and lifestyle features
            4. Overall neighborhood character
            
            Format the summary in clear sections with detailed information.
            """
            
            summary_response = await self.llm_service.get_llm_response(summary_prompt)
            if summary_response["success"]:
                area_data["summary"] = summary_response["response"]
            
            return area_data
            
        except Exception as e:
            logging.error(f"Error getting area data for {zipcode}: {e}")
            return {}

    async def _get_complete_property_data(self, address: str) -> dict:
        """Get comprehensive property data with complete reviews"""
        try:
            property_data = {
                "reviews": [],
                "details": {},
                "nearby": []
            }
            
            # Get detailed property information from Google Maps
            maps_search = GoogleSearch({
                "engine": "google_maps",
                "q": address,
                "type": "place",
                "api_key": self.SERPAPI_KEY,
                "ll": "@42.3600825,-71.0588801,12z"  # Boston coordinates
            })
            place_results = maps_search.get_dict().get("place_results", {})
            
            # Get complete reviews (not truncated)
            if "reviews" in place_results:
                for review in place_results["reviews"]:
                    review_search = GoogleSearch({
                        "engine": "google_maps_reviews",
                        "q": address,
                        "review_id": review.get("review_id"),
                        "api_key": self.SERPAPI_KEY
                    })
                    full_review = review_search.get_dict()
                    if full_review:
                        property_data["reviews"].append(full_review)
            
            # Get LLM summary of property data
            summary_prompt = f"""
            Summarize the following property information for {address}:

            Property Details:
            {json.dumps(place_results, indent=2)}

            Reviews:
            {json.dumps(property_data['reviews'], indent=2)}

            Provide a comprehensive summary that covers:
            1. Property characteristics and condition
            2. Common themes from reviews
            3. Neighborhood context
            4. Investment potential indicators
            
            Format the summary in clear sections with detailed information.
            """
            
            summary_response = await self.llm_service.get_llm_response(summary_prompt)
            if summary_response["success"]:
                property_data["summary"] = summary_response["response"]
            
            return property_data
            
        except Exception as e:
            logging.error(f"Error getting property data for {address}: {e}")
            return {}

    def _chunk_text(self, text: str, metadata: dict) -> List[dict]:
        """Create meaningful chunks from text while preserving context"""
        chunks = []
        
        # Split text into sentences and create overlapping chunks
        sentences = text.split('. ')
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size:
                # Create chunk with current sentences
                chunk_text = '. '.join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {**metadata, **self._get_chunk_context(chunk_text)}
                })
                
                # Start new chunk with overlap
                overlap = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                current_chunk = overlap + [sentence]
                current_length = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '. '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "metadata": {**metadata, **self._get_chunk_context(chunk_text)}
            })
        
        return chunks

    def _get_chunk_context(self, text: str) -> dict:
        return {
            "mentions_schools": str(bool(re.search(r'school|education|student', text, re.I))),
            "mentions_safety": str(bool(re.search(r'safe|crime|security', text, re.I))),
            "mentions_amenities": str(bool(re.search(r'restaurant|park|shop|store', text, re.I))),
            "mentions_transport": str(bool(re.search(r'train|bus|subway|transport', text, re.I))),
            "mentions_price": str(bool(re.search(r'price|value|cost|\$', text, re.I)))
        }

    async def _store_vectors(self, property_row: dict, area_data: dict, property_data: dict):
        """Store properly chunked and contextualized vectors"""
        try:
            vectors = []
            
            # Process area summary
            if "summary" in area_data:
                area_chunks = self._chunk_text(
                    area_data["summary"],
                    {
                        "type": "area_summary",
                        "zipcode": property_row['RegionName'],
                        "source": "llm_summary"
                    }
                )
                
                for chunk in area_chunks:
                    vector = {
                        "id": f"area_{property_row['RegionName']}_{len(vectors)}_{datetime.now().timestamp()}",
                        "values": self.model.encode(chunk["text"]).tolist(),
                        "metadata": {
                            **chunk["metadata"],
                            "text": chunk["text"]
                        }
                    }
                    vectors.append(vector)
            
            # Process property summary
            if "summary" in property_data:
                property_chunks = self._chunk_text(
                    property_data["summary"],
                    {
                        "type": "property_summary",
                        "address": property_row['Address'],
                        "zipcode": property_row['RegionName'],
                        "source": "llm_summary"
                    }
                )
                
                for chunk in property_chunks:
                    vector = {
                        "id": f"property_{property_row['Address']}_{len(vectors)}_{datetime.now().timestamp()}",
                        "values": self.model.encode(chunk["text"]).tolist(),
                        "metadata": {
                            **chunk["metadata"],
                            "text": chunk["text"]
                        }
                    }
                    vectors.append(vector)
            
            # Batch upsert to Pinecone with proper chunking
            if vectors:
                for i in range(0, len(vectors), 100):  # Upload in batches of 100
                    batch = vectors[i:i+100]
                    self.index.upsert(vectors=batch)
                    logging.info(f"Stored batch of {len(batch)} vectors for {property_row['Address']}")
            
        except Exception as e:
            logging.error(f"Error storing vectors: {e}")

    def _format_search_results(self, results: List[dict]) -> str:
        """Format search results for LLM summary"""
        formatted = []
        for result in results:
            formatted.append(f"""
            Title: {result.get('title', '')}
            Snippet: {result.get('snippet', '')}
            Link: {result.get('link', '')}
            """)
        return "\n".join(formatted)

# # Usage example
# if __name__ == "__main__":
#     try:
#         # Initialize the database
#         db = RealEstateVectorDB()
        
#         # Specify the correct absolute path to your CSV file
#         csv_path = os.path.join(
#             os.path.dirname(os.path.dirname(__file__)), 
#             "datasets", 
#             "Property_listings_data_redfin.csv"
#         )
        
#         logging.info(f"Using CSV path: {csv_path}")
        
#         # Load and process the data
#         db.load_property_data(csv_path)
        
#     except Exception as e:
#         logging.error(f"Error in main execution: {str(e)}")