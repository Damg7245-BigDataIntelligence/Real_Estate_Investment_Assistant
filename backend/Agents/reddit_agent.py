from dotenv import load_dotenv
import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from backend.llm_response import reddit_generate_report
 
def pinecone_init():
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".env"))
    load_dotenv(dotenv_path)
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_REDDIT")
 
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "real-estate-assistant"
 
    index = pc.Index(index_name)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(index.describe_index_stats())
    return index, model
 
def search_pinecone_db(input_info, top_k=20):
    """Search for relevant chunks in Pinecone, filtering by multiple years and quarters, and generate a response using Gemini."""
    index, model = pinecone_init()
    queries = {
        "## Market Trends & Pricing Analysis":
            f"What are the current price trends for {input_info['Property Type']} properties in Boston?",
        "## Neighborhood Insights & Budget Match":
            f"Which neighborhoods in Boston match a budget between {input_info['Budget Min']} and {input_info['Budget Max']}?",
        "## Investment Strategies & Expert Advice":
            f"What investment strategies are recommended for {input_info['Investment Goal']} in Boston?",
        "## Demographic Suitability":
            f"Which Boston areas fit the demographic preference of {input_info['Demographic preference']}?"
    }
    print(input_info)
    report_md=""
    for heading, query in queries.items():
        #print(heading, query)
        query_embedding = model.encode([query]).tolist()
        #print(f"Query: {query}\nEmbedding: {query_embedding[:4]}")
        try:
            # Perform a filtered search in Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="reddit"
            )
 
            matches = results.get("matches", [])
            if not matches:
                print("No relevant matches found for the given year-quarter combinations.")
            else:
                # Extract matched texts along with their metadata
                retrieved_data = []
                for match in matches:
                    metadata = match["metadata"]
                    text = metadata.get("text", "N/A")
                    year = metadata.get("year", "Unknown")
                    month = metadata.get("month", "Unknown")
                    title = metadata.get("title", "Untitled")
                    
                    retrieved_data.append((text, year, month, title))
            
                # Create context for Gemini
                context = "\n".join([f"Year: {year}, Month: {month}, Title: {title} - {text}" for text, year, month, title in retrieved_data])
                # Generate response using Gemini if the model is initialized.
                report=reddit_generate_report(context, query, input_info)
                report_md += "\n"+report
        except Exception as e:
            print(f"Error during search: {e}")
            return "Error occurred during search."
    print(report_md)