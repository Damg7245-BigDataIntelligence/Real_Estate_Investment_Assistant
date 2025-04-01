from langgraph.graph import StateGraph, END
from state import RealEstateState, RealEstateReport
from graph_functions import (
    determine_next_node,
    query_preferences,
    property_listings,
    snowflake_analysis,
    websearch_analysis,
    reddit_analysis,
    generate_report,
    initialize_state
)
import asyncio
from typing import Dict, Any

# Global variable to store the compiled graph
_GLOBAL_GRAPH = None

def sync_query_preferences(state):
    """Synchronous wrapper for async query_preferences"""
    try:
        # Create new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the async function
        result = loop.run_until_complete(query_preferences(state))
        
        # Clean up
        loop.close()
        
        return result
    except Exception as e:
        print(f"Error in sync_query_preferences: {str(e)}")
        state["error"] = str(e)
        return state

def initialize_real_estate_graph():
    """
    Initialize the real estate investment research graph once and store it in memory.
    Returns the compiled graph instance.
    """
    global _GLOBAL_GRAPH
    
    if _GLOBAL_GRAPH is None:
        print("Initializing new real estate investment research graph...")
        
        # Create a new graph
        graph = StateGraph(RealEstateState)
        
        # Add nodes - all synchronous now
        graph.add_node("query_preferences", query_preferences)
        graph.add_node("property_listings", property_listings)
        graph.add_node("snowflake_analysis", snowflake_analysis)
        graph.add_node("websearch_analysis", websearch_analysis)
        graph.add_node("reddit_analysis", reddit_analysis)
        graph.add_node("generate_report", generate_report)
        
        # Set the entry point
        graph.set_entry_point("query_preferences")
        
        # Add conditional edges using the router function
        graph.add_conditional_edges(
            "query_preferences",
            determine_next_node,
            {
                "property_listings": "property_listings",
                "snowflake_analysis": "snowflake_analysis",
                "websearch_analysis": "websearch_analysis",
                "reddit_analysis": "reddit_analysis",
                "generate_report": "generate_report",
                "END": END
            }
        )
        
        graph.add_conditional_edges(
            "property_listings",
            determine_next_node,
            {
                "snowflake_analysis": "snowflake_analysis",
                "websearch_analysis": "websearch_analysis",
                "reddit_analysis": "reddit_analysis",
                "generate_report": "generate_report",
                "END": END
            }
        )
        
        graph.add_conditional_edges(
            "snowflake_analysis",
            determine_next_node,
            {
                "websearch_analysis": "websearch_analysis",
                "reddit_analysis": "reddit_analysis",
                "generate_report": "generate_report",
                "END": END
            }
        )
        
        graph.add_conditional_edges(
            "websearch_analysis",
            determine_next_node,
            {
                "reddit_analysis": "reddit_analysis",
                "generate_report": "generate_report",
                "END": END
            }
        )
        
        graph.add_conditional_edges(
            "reddit_analysis",
            determine_next_node,
            {
                "generate_report": "generate_report",
                "END": END
            }
        )
        
        graph.add_conditional_edges(
            "generate_report",
            determine_next_node,
            {
                "END": END
            }
        )
        
        # Compile and store the graph
        _GLOBAL_GRAPH = graph.compile()
        print("Real estate investment research graph initialized and compiled successfully")
    
    return _GLOBAL_GRAPH

def run_real_estate_graph(query: str, preferences: Dict[str, Any] = None):
    """Execute the workflow for a user query using the initialized research graph."""
    print("\n" + "#" * 100)
    print(f"📊 STARTING RESEARCH GRAPH EXECUTION 📊")
    print("#" * 100 + "\n")

    try:
        # Initialize state with report file
        state = initialize_state(query)
        print(f"Initial state with report file: {state.get('report_file')}")
        
        if preferences:
            state["user_preferences"] = preferences
        
        # Store the original report file path to return later
        report_file_path = state.get("report_file")
        
        print(f"State after report initialization: {state}")
        
        # Get the pre-initialized graph
        graph = initialize_real_estate_graph()
        
        # Run the workflow using graph's invoke method
        try:
            result = graph.invoke(state)
            
            # Extract output from result and include report_file path
            return {
                "output": result.get("output", "No output generated."),
                "error": result.get("error"),
                "report_file": report_file_path  # Add this to return the file path
            }
            
        except Exception as e:
            print(f"Error during graph execution: {e}")
            return {
                "output": "An error occurred during processing.",
                "error": str(e),
                "report_file": report_file_path  # Still return the file path even if there's an error
            }

    except Exception as e:
        print(f"Error in run_real_estate_graph: {e}")
        return {
            "output": "An error occurred while setting up the research task.",
            "error": str(e),
            "report_file": None  # No file path in case of early error
        } 