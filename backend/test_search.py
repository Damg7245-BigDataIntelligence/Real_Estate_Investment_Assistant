from backend.Pipeline.pinecone_manager import RealEstateVectorDB
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_search():
    try:
        # Initialize the vector DB
        db = RealEstateVectorDB()
        # db.delete_index("real-estate-reviews")
        # Get user input
        print("\n=== Real Estate Search ===")
        print("Example questions you can ask:")
        print("- Which areas have good schools?")
        print("- What are safe neighborhoods with low crime rates?")
        print("- Tell me about family-friendly areas")
        print("- Where can I find properties with good investment potential?")
        
        query = input("\nEnter your question: ")
        zipcode = input("Enter zipcode (optional, press Enter to skip): ").strip()
        
        # Search properties
        zipcode = zipcode if zipcode else None
        results = db.search_properties(query=query, zipcode=zipcode)
        
        # Display results
        print("\n=== Search Results ===")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Score: {result['score']:.3f}")
            print(f"Type: {result['metadata']['type']}")
            
            if result['metadata']['type'] == 'property':
                print(f"Address: {result['metadata']['address']}")
                print(f"ZIP Code: {result['metadata']['zipcode']}")
            
            print(f"Content: {result['metadata']['text'][:200]}...")  # Show first 200 chars
            print("-" * 80)

    except Exception as e:
        logging.error(f"Error during search: {e}")

if __name__ == "__main__":
    test_search() 