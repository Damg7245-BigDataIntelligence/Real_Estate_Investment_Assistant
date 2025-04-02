from langgraph.graph import StateGraph, END  # Main graph library components
from backend.state import ResearchState
from backend.graph_functions import (
    run_oracle, router, reddit_search, web_search,
    snowflake_search, property_search, demographics_search, review_search, 
    generate_final_answer
)

# Global variable to store the compiled graph (singleton pattern)
_GLOBAL_GRAPH = None

def initialize_research_graph():
    """Builds and configures the workflow graph (only once)"""
    print("Initializing new research graph...")
    global _GLOBAL_GRAPH  # Access the global graph variable
    
    if _GLOBAL_GRAPH is None:
        print("Initializing new research graph...")
    
        # Create empty graph container
        # Uses our custom ResearchState to track progress
        graph = StateGraph(ResearchState)

        # Add all workflow nodes (processing steps)
        # Each node wraps a function that modifies the state
        # Add all nodes
        graph.add_node("oracle", lambda x: run_oracle(x))
        graph.add_node("reddit_search", lambda x: reddit_search(x))
        graph.add_node("web_search", lambda x: web_search(x))
        graph.add_node("demographics_search", lambda x: demographics_search(x))
        graph.add_node("property_search", lambda x: property_search(x))
        graph.add_node("snowflake_search", lambda x: snowflake_search(x))
        graph.add_node("review_search", lambda x: review_search(x))
        graph.add_node("final_answer", lambda x: generate_final_answer(x))

        # Add join nodes for branch coordination
        graph.add_node("demographics_merge", lambda x: x)  # Merges property/snowflake branches
        graph.add_node("main_merge", lambda x: x)          # Merges all main branches

        # Set starting point - first node to execute
        graph.set_entry_point("oracle")

        # Main parallel branches (Level 1)
        graph.add_conditional_edges(
            "oracle",
            lambda _: "main_parallel",
            {
                "main_parallel": {
                    "social_analysis": "reddit_search",
                    "web_analysis": "web_search",
                    "demographics_analysis": "demographics_search"
                }
            }
        )

        # Demographics sub-branches (Level 2)
        graph.add_conditional_edges(
            "demographics_search",
            lambda _: "demographics_parallel",
            {
                "demographics_parallel": {
                    "property_review_chain": "property_search",
                    "snowflake_analysis": "snowflake_search"
                }
            }
        )

        # Configure property review chain (Sequential)
        graph.add_edge("property_search", "review_search")
        graph.add_edge("review_search", "demographics_merge")
        
        # Connect snowflake to merge point
        graph.add_edge("snowflake_search", "demographics_merge")

        # Merge all branches
        graph.add_edge("reddit_search", "main_merge")
        graph.add_edge("web_search", "main_merge")
        graph.add_edge("demographics_merge", "main_merge")
        
        # Connect to final answer
        graph.add_edge("main_merge", "final_answer")
        graph.add_edge("final_answer", END)

        _GLOBAL_GRAPH = graph.compile()
    
    return _GLOBAL_GRAPH

def run_research_graph(query):
    """Executes the workflow for a user query using the initialized research graph."""
    # Print execution header for debugging
    print("\n" + "#" * 100)
    print(f"📊 STARTING RESEARCH GRAPH EXECUTION 📊")
    print("#" * 100 + "\n")

    # Set up initial state - like a shared whiteboard for all nodes
    state = {
        "input": query,  # User's original question
        "chat_history": [],  # Conversation history (for chat contexts)
        "intermediate_steps": [],  # Stores search results and decisions

    }

    # Get the pre-built workflow graph
    graph = initialize_research_graph()
    print(f"Using existing graph with nodes: {list(graph.nodes.keys())}")

    # Run the workflow with our initial state
    try:
        result = graph.invoke(state)
    except Exception as e:
        print(f"Error during graph execution: {e}")
        return "An error occurred while processing your request. Please try again."

    # Completion notification
    print("\n" + "#" * 100)
    print("📋 GRAPH EXECUTION COMPLETED")
    print("#" * 100 + "\n")

    # First try: Get direct answer from output field
    if "output" in result:
        print("Using 'output' field from result")
        return result["output"]

    # Fallback: Check intermediate steps for answer
    try:
        if "intermediate_steps" in result and result["intermediate_steps"]:
            for step in result["intermediate_steps"]:
                # Look for final answer in execution history
                if hasattr(step, 'tool') and step.tool == "final_answer_result":
                    print("Using 'final_answer_result' from intermediate_steps")
                    return step.log  # Return the stored answer
    except Exception as e:
        print(f"Error extracting from intermediate_steps: {e}")

    # If all else fails, return error message
    print("No result found, returning fallback message")
    return "No comprehensive results available. Please try again with a different query."


