from nltk.tokenize import sent_tokenize
import os
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import requests
from urllib.parse import urlparse
from datetime import datetime, timezone  # in case you need to convert created_utc to datetime

def sentence_chunks(text: str) -> list[str]:
    max_sentence_length = 256
    sentences = sent_tokenize(text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If a single sentence exceeds the max length, handle it separately
        if len(sentence) > max_sentence_length:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(sentence.strip())
        # If adding the sentence (with a space if needed) stays within the limit
        elif len(current_chunk) + len(sentence) + (1 if current_chunk else 0) <= max_sentence_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        # Otherwise, finalize the current chunk and start a new one
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            
    # Append any leftover text as a final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print("Number of chunks: ",len(chunks))
    return chunks


class PineconeInsertion:
    def __init__(self):
        # Configure Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # Load environment variables
        dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
        load_dotenv(dotenv_path)
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index_name = "real-estate-assistant"
        self.dimension = 384  # Matching the embedding model's output size
        
        # Check and create Pinecone index if it doesn’t exist
        if self.index_name not in [index["name"] for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logging.info(f"Index '{self.index_name}' created.")
        else:
            logging.info(f"Index '{self.index_name}' already exists.")
        
        # Connect to the index and print its stats
        self.index = self.pc.Index(self.index_name)
        logging.info(f"Pinecone index stats: {self.index.describe_index_stats()}")
        
        # Load Sentence Transformer Model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logging.info("Sentence Transformer model loaded.")
        

    def insert_embeddings(self, post_data):
        """Processes markdown from a presigned URL, generates embeddings, and inserts into Pinecone."""
        try:
            text = post_data["selftext"]
            # Process chunks from markdown content. Note that sentence_chunks now returns a list of strings.
            chunks = sentence_chunks(text)
            if not chunks:
                logging.warning("No chunks extracted. Skipping embedding.")
                return

            # Use chunks directly (each chunk is a string)
            embeddings = self.model.encode(chunks).tolist()
            print(f"Generated embeddings for {len(embeddings)} chunks.")

            # Prepare batch upserts for Pinecone
            pinecone_data = []
            # Ensure created_utc is a datetime object; if not, convert it.
            # For example, if post_data["created_utc"] is a timestamp, convert as follows:
            if not isinstance(post_data["created_utc"], datetime):
                created_dt = datetime.fromtimestamp(post_data["created_utc"], tz=timezone.utc)
            else:
                created_dt = post_data["created_utc"]

            quarter = (created_dt.month - 1) // 3 + 1
            year = created_dt.year
            print(created_dt, year, quarter)
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "text": chunk,  # chunk data
                    "title": post_data["title"],
                    "created_utc": str(created_dt),  # store as ISO formatted string
                    "year": year,
                    "quarter": quarter,
                    "month": created_dt.month,
                    "subreddit": post_data["subreddit"],
                    "post_id": post_data["post_id"]
                }
                pinecone_data.append((f"{post_data["post_id"]}_{year}_{created_dt.month}_{i}", embedding, metadata))
            
            # Insert data into Pinecone in batch
            self.index.upsert(pinecone_data, namespace="reddit")
            print(f"Inserted {len(pinecone_data)} chunks into Pinecone successfully.")
        except Exception as e:
            print(f"Error processing data: {e}")
        
#     def search_pinecone_db(self, query, year_quarter_dict, top_k=20):
#         """Search for relevant chunks in Pinecone, filtering by multiple years and quarters, and generate a response using Gemini."""
#         query_embedding = self.model.encode([query]).tolist()
#         try:
#             # Construct metadata filter for multiple years and quarters.
#             # Changed to use numeric comparisons since 'year' and 'quarter' are stored as numbers.
#             filter_criteria = {
#                 "$or": [
#                     {"year": {"$eq": year}, "quarter": {"$in": quarters}}
#                     for year, quarters in year_quarter_dict.items()
#                 ]
#             }
#             logging.info(f"Filter criteria: {filter_criteria}")

#             # Perform a filtered search in Pinecone
#             results = self.index.query(
#                 vector=query_embedding,
#                 top_k=top_k,
#                 include_metadata=True,
#                 filter=filter_criteria  # Apply filtering
#             )

#             matches = results.get("matches", [])
#             if not matches:
#                 logging.warning("No relevant matches found for the given year-quarter combinations.")
#                 return "No relevant information found for the specified year and quarters."

#             # Extract matched texts along with their metadata
#             retrieved_data = [(match["metadata"]["text"], match["metadata"]["year"], match["metadata"]["quarter"]) for match in matches]
            
#             # Create context for Gemini
#             context = "\n".join([f"Year: {year}, Quarter: {quarter} - {text}" for text, year, quarter in retrieved_data])
#             prompt = f"""You are an AI assistant tasked with analyzing Nvidia's financial data. 
# Below is relevant financial information retrieved from a vector database, with each entry associated with a specific year and quarter. 
# Use this context to answer the question accurately.
# Question: {query}
# Context: {context}
# """
#             # Generate response using Gemini if the model is initialized.
#             if self.gemini_model is not None:
#                 response = self.gemini_model.generate_content(prompt)
#                 return response.text
#             else:
#                 logging.error("Gemini model is not initialized.")
#                 return "Gemini model is not initialized."
#         except Exception as e:
#             logging.error(f"Error during search: {e}")
#             return "Error occurred during search."
