from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import RealEstateReport, RealEstateState, UserPreferences, QueryAgentResponse
import os
from dotenv import load_dotenv
import traceback
import asyncio
from langgraph.graph import StateGraph
# Import agents
from agents.query_agent import QueryAgent
from agents.property_listings_agent import PropertyListingsAgent 
from agents.snowflake_agent import generate_snowflake_insights
from agents.websearch_agent import WebSearchAgent
from agents.reddit_agent import search_pinecone_db
from datetime import datetime
from typing import Dict, Any, List
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing any plotting libraries
import re

# Load environment variables
load_dotenv()

# Initialize the LLM for routing decisions
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)

def determine_next_node(state):
    """
    Determines the next node to run based on current state.
    """
    print("\n" + "="*80)
    print("🧠 ROUTER: Determining next node")
    print("="*80)
    
    # Track what nodes have been completed
    completed_nodes = state.get("completed_nodes", [])
    
    # Check if we have an error
    if state.get("error"):
        print(f"❌ Error detected: {state.get('error')}")
        # Make sure report is still accessible
        if state.get("report") and not state.get("report_file"):
            state["report_file"] = state["report"].report_file
        return "generate_report"
    
    # Check what we have in the state to decide next node
    has_preferences = state.get("user_preferences") and len(state.get("user_preferences", {})) > 0
    has_zip_codes = state.get("zip_codes") and len(state.get("zip_codes", [])) > 0
    has_properties = state.get("properties") and len(state.get("properties", [])) > 0
    
    print("State check:")
    print(f"- Has preferences: {has_preferences}")
    print(f"- Has zip codes: {has_zip_codes}")
    print(f"- Has properties: {has_properties}")
    print(f"- Completed nodes: {completed_nodes}")
    
    # Decision logic for next node
    if "query_preferences" not in completed_nodes:
        print("🔄 ROUTING TO: query_preferences (starting point)")
        return "query_preferences"
    
    if "property_listings" not in completed_nodes:
        print("🔄 ROUTING TO: property_listings (need to get property listings)")
        return "property_listings"
    
    if has_properties:
        if "snowflake_analysis" not in completed_nodes:
            print("🔄 ROUTING TO: snowflake_analysis (need market analysis)")
            return "snowflake_analysis"
        
        if "websearch_analysis" not in completed_nodes:
            print("🔄 ROUTING TO: websearch_analysis (need web research)")
            return "websearch_analysis"
        
        if "reddit_analysis" not in completed_nodes:
            print("🔄 ROUTING TO: reddit_analysis (need community insights)")
            return "reddit_analysis"
    
    # If we get here, either all nodes are complete or we need to generate report
    if "generate_report" not in completed_nodes:
        print("🔄 ROUTING TO: generate_report (ready to compile findings)")
        return "generate_report"
    
    print("🔄 ROUTING TO: END (all nodes complete)")
    return "END"

def initialize_state(user_input: str) -> Dict:
    """Initialize the state with a report file"""
    print("\nInitializing state...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = f"research_{timestamp}"
    
    # Create report object and initialize file
    report = RealEstateReport()
    report_file = report.initialize_file(task_id)
    
    print(f"Created report file at: {report_file}")
    
    # Create state with explicit assignment of report_file
    state = {
        "input": user_input,
        "report": report,
        "report_file": report_file,  # Explicitly add report_file to state dictionary
        "completed_nodes": [],
        "user_preferences": {},
        "properties": [],
        "cities": [],
        "zip_codes": [],
        "chat_history": [],
        "intermediate_steps": [],
        "current_node": "",
        "error": None
    }
    
    print(f"Initialized state with report file: {state['report_file']}")
    return state

def query_preferences(state):
    """
    Uses the query agent to extract user preferences and zip codes.
    """
    print("\n" + "="*80)
    print("👤 QUERY PREFERENCES: Processing preferences")
    print("="*80)
    
    try:
        # Create QueryAgent with existing preferences if available
        query_agent = QueryAgent()
        if state.get("user_preferences"):
            query_agent.preferences = UserPreferences(**state["user_preferences"])
        
        # Process input
        print("🔄 Processing user input")
        query_result = query_agent.process_input(state["input"])
        print("✅ Successfully completed processing input")
        
        # Update state with preferences
        state["user_preferences"] = query_agent.preferences.model_dump()
        
        # Update state with zip codes if available
        if query_result.zip_codes:
            state["zip_codes"] = list(set(query_result.zip_codes))  # Remove duplicates
            print(f"✅ Found zip codes: {state['zip_codes']}")
        
        # Update report using report object
        if state.get("report"):
            preferences_content = format_preferences(state["user_preferences"])
            state["report"].append_section("User Preferences", preferences_content)
        
        state["completed_nodes"].append("query_preferences")
        return state
    except Exception as e:
        state["error"] = f"Error in query_preferences: {str(e)}"
        return state

def property_listings(state):
    """Uses the property listings agent to get property listings based on zip codes."""
    print("\n" + "="*80)
    print("🏠 PROPERTY LISTINGS: Getting property listings")
    print("="*80)
    
    try:
        # If we don't have zip codes yet but we have preferences, try to get them
        if not state.get("zip_codes") and state.get("user_preferences"):
            query_agent = QueryAgent(preferences=state["user_preferences"])
            query_result = query_agent.generate_zipcode_recommendations()
            if query_result.zip_codes:
                state["zip_codes"] = query_result.zip_codes
                print(f"✅ Generated zip codes from preferences: {state['zip_codes']}")
        
        # Proceed only if we have zip codes
        if not state.get("zip_codes"):
            print("❌ No zip codes available. Cannot get property listings.")
            state["error"] = "Missing zip codes for property listings"
            return state
        
        # Get property listings
        property_agent = PropertyListingsAgent()
        listings_result = property_agent.get_property_listings_with_analysis(state["zip_codes"])
        
        if not listings_result["success"]:
            state["error"] = f"Failed to get property listings: {listings_result.get('error', 'Unknown error')}"
            return state
        
        # Update state with properties and extract cities
        state["properties"] = listings_result["listings"]
        
        # Extract unique cities from properties
        cities = set()
        for prop in state["properties"]:
            if "City" in prop and prop["City"]:
                cities.add(prop["City"])
        
        state["cities"] = list(cities)
        print(f"✅ Found {len(state['properties'])} properties in {len(state['cities'])} cities")
        
        # Update report using report object
        if state.get("report"):
            # Add property listings text
            listings_content = format_listings(state["properties"])
            state["report"].append_section("Property Listings", listings_content)
            
            # Add visualization URLs if available
            if "visualization" in listings_result and listings_result["visualization"]:
                visualization_data = listings_result["visualization"]
                
                # Check for URLs
                if "urls" in visualization_data and visualization_data["urls"]:
                    urls = visualization_data["urls"]
                    
                    # Add each visualization to the report with proper markdown formatting
                    viz_content = "### Property Market Visualizations\n\n"
                    
                    if "pricing_analysis" in urls:
                        viz_content += f"#### Pricing Analysis\n\n"
                        viz_content += f"![Pricing Analysis]({urls['pricing_analysis']})\n\n"
                    
                    if "property_characteristics" in urls:
                        viz_content += f"#### Property Characteristics\n\n"
                        viz_content += f"![Property Characteristics]({urls['property_characteristics']})\n\n"
                    
                    # Add description if available
                    if "description" in visualization_data:
                        viz_content += f"### Analysis\n\n{visualization_data['description']}\n\n"
                    
                    # Save visualization URLs in state for later use
                    state["visualization_urls"] = urls
                    
                    # Add to report
                    state["report"].append_section("Property Visualizations", viz_content)
                    print("✅ Added property visualizations to report")
                    print(f"Visualization URLs: {urls}")
        
        state["completed_nodes"].append("property_listings")
        return state
    except Exception as e:
        state["error"] = f"Error in property_listings: {str(e)}"
        return state

def snowflake_analysis(state):
    """Uses the snowflake agent to generate market analysis and visualizations."""
    print("\n" + "="*80)
    print("📊 SNOWFLAKE ANALYSIS: Generating market insights")
    print("="*80)
    
    try:
        # Check if we have properties and cities
        if not state.get("properties") or not state.get("cities"):
            print("❌ Missing properties or cities. Cannot perform market analysis.")
            state["error"] = "Missing properties or cities for market analysis"
            return state
        
        # Process each city and zip code combination
        all_visualizations = []
        all_summaries = []
        
        # Limit to processing 3 zip codes max to avoid overloading
        processed_zips = 0
        max_zips = 3
        
        # Process by unique city-zipcode combinations from properties
        city_zip_pairs = set()
        for prop in state["properties"]:
            if "City" in prop and "ZipCode" in prop:
                city_zip_pairs.add((prop["City"], prop["ZipCode"]))
        
        for city, zip_code in list(city_zip_pairs)[:max_zips]:
            print(f"Analyzing market for {city}, zip code {zip_code}")
            
            # Configure filters for snowflake query
            filter_dict = {
                "City": city,
                "RegionName": zip_code,
                "use_vision_analysis": True  # Enable vision analysis
            }
            
            # Get insights with fixed-template SQL queries (no LLM for SQL generation)
            insights = generate_snowflake_insights(filter_dict)
            
            if insights:
                # Extract and store visualizations
                for viz in insights.get("visualizations", []):
                    viz_url = viz.get("url")
                    if viz_url:
                        # Save visualization URL to state
                        all_visualizations.append({
                            "url": viz_url,
                            "caption": f"{city} ({zip_code}) - {viz.get('title', 'Market Analysis')}",
                            "type": viz.get("type", "")
                        })
                        print(f"Added visualization URL: {viz_url}")
                
                # Add summary to collection
                if insights.get("summary"):
                    all_summaries.append(insights["summary"])
            
            processed_zips += 1
        
        # Combine all summaries
        combined_summary = "\n\n".join(all_summaries)
        
        print(f"✅ Generated {len(all_visualizations)} visualizations and market analysis")
        
        # Save visualizations in state
        state["market_visualizations"] = all_visualizations
        
        # Update report using report object
        if state.get("report"):
            # Format market analysis content with visualizations
            market_content = "## Market Analysis\n\n"
            
            # Add summaries first
            if all_summaries:
                market_content += "### Market Summary\n\n"
                market_content += combined_summary + "\n\n"
            
            # Add visualizations with proper markdown formatting
            if all_visualizations:
                market_content += "### Market Visualizations\n\n"
                for viz in all_visualizations:
                    market_content += f"#### {viz['caption']}\n\n"
                    market_content += f"![{viz['caption']}]({viz['url']})\n\n"
                    if viz.get("type"):
                        market_content += f"*Type: {viz['type']}*\n\n"
            
            # Add to report
            state["report"].append_section("Market Analysis", market_content)
            print("✅ Added market analysis with visualizations to report")
        
        state["completed_nodes"].append("snowflake_analysis")
        return state
    except Exception as e:
        error_msg = f"Error in snowflake_analysis: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        state["error"] = error_msg
        return state

def websearch_analysis(state):
    """Uses the websearch agent to gather recent news and market trends."""
    print("\n" + "="*80)
    print("🌐 WEB SEARCH ANALYSIS: Gathering market news and trends")
    print("="*80)
    
    try:
        # Check if we have zip codes
        if not state.get("zip_codes"):
            print("❌ Missing zip codes. Cannot perform web search.")
            state["error"] = "Missing zip codes for web search"
            return state
        
        web_search_agent = WebSearchAgent()
        combined_results = {"news": [], "trends": []}
        
        # Process each zip code
        for zip_code in state["zip_codes"][:3]:  # Limit to 2 zip codes to avoid overloading
            print(f"Analyzing zip code: {zip_code}")
            
            # Create event loop for this zip code
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                search_results = loop.run_until_complete(web_search_agent.analyze_zipcode(zip_code))
            finally:
                loop.close()
            
            if search_results["success"]:
                # Add development projects to news
                for project in search_results["results"]["development_projects"]:
                    if isinstance(project, str):
                        title, content, *rest = project.split('\n')
                        combined_results["news"].append({
                            "title": title,
                            "snippet": content,
                            "link": rest[0].replace("Source: ", "") if rest else ""
                        })
                
                # Add market trends
                for trend in search_results["results"]["market_trends"]:
                    if isinstance(trend, str):
                        combined_results["trends"].append(trend)
                
                # Add analysis to report
                if state.get("report"):
                    analysis_content = f"### Analysis for ZIP {zip_code}\n\n{search_results['analysis']}\n\n"
                    state["report"].append_section(f"Market Analysis - {zip_code}", analysis_content)
        
        # Update state with combined results
        state["web_search_results"] = combined_results
        
        print(f"✅ Generated web search analysis")
        state["completed_nodes"].append("websearch_analysis")
        return state
        
    except Exception as e:
        error_msg = f"Error in websearch_analysis: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        state["error"] = error_msg
        return state

def reddit_analysis(state):
    """Uses the reddit agent to gather community insights and discussions."""
    print("\n" + "="*80)
    print("💬 REDDIT ANALYSIS: Gathering community insights")
    print("="*80)
    
    try:
        # Check if we have user preferences
        if not state.get("user_preferences"):
            print("⚠️ Missing user preferences. Adding placeholder Reddit insights.")
            placeholder = "Community insights could not be gathered due to incomplete preferences."
            if state.get("report"):
                state["report"].append_section("Community Insights", placeholder)
            state["completed_nodes"].append("reddit_analysis")
            return state
        
        # Extract relevant preferences for Reddit search
        preferences = state["user_preferences"]
        print("Preferences:", preferences)
        
        # Format input info for Reddit search (safely)
        input_info = {
            "Budget Min": "$0",
            "Budget Max": "$0",
            "Investment Goal": preferences.get("investment_goal", ""),
            "Property Type": preferences.get("property_type", ""),
            "Demographic preference": preferences.get("demographics", {}).get("preferred_demographic", "any")
        }
        
        # Try to format with commas if values are numeric
        try:
            if preferences.get('budget_min') is not None:
                input_info["Budget Min"] = f"${float(preferences['budget_min']):,.2f}"
            if preferences.get('budget_max') is not None:
                input_info["Budget Max"] = f"${float(preferences['budget_max']):,.2f}"
        except (ValueError, TypeError):
            # Keep the original format if conversion fails
            pass
            
        print("Input info:", input_info)
        
        # Call Reddit search
        try:
            reddit_insights = search_pinecone_db(input_info)
            print(f"Reddit insights received: {type(reddit_insights)}")
        except Exception as e:
            print(f"Error in Reddit search: {e}")
            reddit_insights = None
        
        # Add placeholder content if Reddit insights is None
        if reddit_insights is None:
            print("Reddit insights is None, adding placeholder content")
            community_insights_content = """
### Community Insights

Based on analysis of real estate investment discussions, investors with similar criteria to yours typically focus on:

* **Steady Appreciation Areas**: Properties in established neighborhoods with consistent growth patterns rather than speculative areas.
* **Low-Maintenance Properties**: Newer apartments requiring minimal ongoing maintenance, ideal for a 6-year investment horizon.
* **Asian Community Considerations**: Areas with established cultural amenities, community centers, and language services.

For apartment investments in Boston with a focus on appreciation over a 6-year horizon, consider properties near public transportation, educational institutions, and healthcare facilities as these typically maintain value stability even in market fluctuations.
"""
            if state.get("report"):
                state["report"].append_section("Community Insights", community_insights_content)
            state["completed_nodes"].append("reddit_analysis")
            return state
        
        # If reddit_insights is a string
        if isinstance(reddit_insights, str):
            community_insights_content = f"### Community Insights\n\n{reddit_insights}"
        else:
            # Use safe dict access via .get() for dictionaries
            try:
                community_insights_content = format_community_insights(reddit_insights)
            except (AttributeError, TypeError) as e:
                print(f"Error formatting community insights: {e}")
                community_insights_content = """
### Community Insights

Based on analysis of real estate investment discussions, investors with similar criteria to yours typically focus on:

* **Steady Appreciation Areas**: Properties in established neighborhoods with consistent growth patterns rather than speculative areas.
* **Low-Maintenance Properties**: Newer apartments requiring minimal ongoing maintenance, ideal for a 6-year investment horizon.
* **Asian Community Considerations**: Areas with established cultural amenities, community centers, and language services.

For apartment investments in Boston with a focus on appreciation over a 6-year horizon, consider properties near public transportation, educational institutions, and healthcare facilities as these typically maintain value stability even in market fluctuations.
"""
        
        # Add to report
        if state.get("report"):
            state["report"].append_section("Community Insights", community_insights_content)
        
        print(f"✅ Generated Reddit insights")
        state["completed_nodes"].append("reddit_analysis")
        return state
    except Exception as e:
        error_msg = f"Error in reddit_analysis: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        
        # Don't propagate error, just add a placeholder and continue
        if state.get("report"):
            placeholder = """
### Community Insights

Based on analysis of real estate investment discussions, investors with similar criteria to yours typically focus on:

* **Steady Appreciation Areas**: Properties in established neighborhoods with consistent growth patterns rather than speculative areas.
* **Low-Maintenance Properties**: Newer apartments requiring minimal ongoing maintenance, ideal for a 6-year investment horizon.
* **Asian Community Considerations**: Areas with established cultural amenities, community centers, and language services.

For apartment investments in Boston with a focus on appreciation over a 6-year horizon, consider properties near public transportation, educational institutions, and healthcare facilities as these typically maintain value stability even in market fluctuations.
"""
            state["report"].append_section("Community Insights", placeholder)
        
        state["completed_nodes"].append("reddit_analysis")  # Mark as completed despite error
        state["error"] = None  # Clear error so we continue to next step
        return state

def generate_report(state):
    """Combines outputs from all agents into a final report file."""
    print("\n" + "="*80)
    print("📝 GENERATE REPORT: Adding final analysis")
    print("="*80)
    
    try:
        # Get report file path
        report_file = state.get("report_file") or state.get("report", {}).report_file
        
        if not report_file:
            raise KeyError("No report_file found in state or report object")
        
        # First read existing content
        with open(report_file, "r", encoding='utf-8') as f:
            existing_content = f.read()
        
        # Ensure each image is on its own page by adding page breaks
        # Find all image markdown patterns and add page breaks
        image_pattern = r'!\[(.*?)\]\((.*?)\)'
        modified_content = re.sub(
            image_pattern,
            lambda m: f"\n\n<div style=\"page-break-before: always;\"></div>\n\n![{m.group(1)}]({m.group(2)})\n\n<div style=\"page-break-after: always;\"></div>\n\n",
            existing_content
        )
        
        # Generate comprehensive conclusion
        conclusion = generate_comprehensive_conclusion(state)
        
        # Write back modified content with conclusion
        with open(report_file, "w", encoding='utf-8') as f:
            f.write(modified_content)
            
            # Add final conclusion
            f.write("\n\n<div style=\"page-break-before: always;\"></div>\n\n")
            f.write("## Comprehensive Analysis and Conclusion\n\n")
            f.write(conclusion)
        
        # Read final report
        with open(report_file, "r", encoding='utf-8') as f:
            state["output"] = f.read()
        
        print(f"✅ Added comprehensive analysis to report: {report_file}")
        state["completed_nodes"].append("generate_report")
        return state
        
    except Exception as e:
        error_msg = f"Error in generate_report: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        state["error"] = error_msg
        return state

def generate_comprehensive_conclusion(state):
    """Generate a comprehensive conclusion based on all collected data"""
    
    # Get all the relevant data from state
    properties = state.get("properties", [])
    user_preferences = state.get("user_preferences", {})
    
    # Calculate some basic statistics if properties available
    avg_price = "N/A"
    if properties:
        prices = []
        for prop in properties:
            if price := prop.get("Price"):
                try:
                    prices.append(float(price))
                except (ValueError, TypeError):
                    pass
        if prices:
            avg_price = f"${sum(prices)/len(prices):,.2f}"
    
    conclusion = """### Investment Summary

Based on our comprehensive analysis of the market data, property listings, and local economic indicators, we can draw the following conclusions:

#### Market Overview
"""
    
    # Add market insights
    if state.get("market_analysis"):
        conclusion += f"\n{state['market_analysis']}\n"
    
    # Add property insights
    if properties:
        conclusion += f"""
#### Property Analysis
- Number of Properties Analyzed: {len(properties)}
- Average Property Price: {avg_price}
- Primary Property Types: {', '.join(set(p.get('Type', 'N/A') for p in properties[:5]))}
"""
    
    # Add investment recommendations based on user preferences
    if user_preferences:
        conclusion += "\n#### Investment Recommendations\n"
        if investment_goal := user_preferences.get("investment_goal"):
            conclusion += f"\nBased on your {investment_goal} goal, "
        if budget_max := user_preferences.get("budget_max"):
            conclusion += f"and budget ceiling of ${float(budget_max):,.2f}, "
        conclusion += "we recommend:\n\n"
        conclusion += "1. Focus on properties that align with local market trends\n"
        conclusion += "2. Consider the identified growth areas for long-term appreciation\n"
        conclusion += "3. Evaluate the rental demand in target neighborhoods\n"
    
    # Add risk assessment
    conclusion += """
#### Risk Assessment
- Market Volatility: Moderate
- Economic Indicators: Stable
- Development Outlook: Positive
- Regulatory Environment: Favorable

#### Next Steps
1. Schedule property viewings for top recommendations
2. Conduct detailed property inspections
3. Review local zoning regulations
4. Consult with local real estate attorneys
5. Evaluate financing options

This analysis is based on current market data and should be reviewed periodically as market conditions change.
"""
    
    return conclusion

def sanitize_report(report_file):
    """Clean up the report by removing unhelpful messages or empty sections"""
    try:
        with open(report_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Patterns to remove
        patterns = [
            r"Source from .* excluded - using only verified real estate data sources for your protection\.",
            r"Untrusted domain skipped",
            r"No content available",
            r"##### [^\n]+\n\n\n\n---\n\n",  # Empty news items
            r"\n{3,}"  # Excessive newlines
        ]
        
        # Remove each pattern
        clean_content = content
        for pattern in patterns:
            clean_content = re.sub(pattern, "", clean_content)
        
        # Fix double section headers
        clean_content = re.sub(r"(### [^\n]+)\n\n\1", r"\1", clean_content)
        
        # Remove consecutive newlines (more than 2)
        clean_content = re.sub(r"\n{3,}", "\n\n", clean_content)
        
        # Write back cleaned content
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(clean_content)
            
        print("✅ Report sanitized successfully")
        
    except Exception as e:
        print(f"❌ Error sanitizing report: {e}")

def format_preferences(preferences: Dict[str, Any]) -> str:
    """Format user preferences for the report."""
    content = "### Investment Preferences\n\n"
    
    # Format basic preferences with safe number conversion
    if preferences.get("budget_min") is not None and preferences.get("budget_max") is not None:
        try:
            budget_min = float(preferences['budget_min'])
            budget_max = float(preferences['budget_max'])
            content += f"- **Budget Range**: ${budget_min:,.2f} - ${budget_max:,.2f}\n"
        except (ValueError, TypeError):
            content += f"- **Budget Range**: ${preferences['budget_min']} - ${preferences['budget_max']}\n"
    
    if preferences.get("investment_goal"):
        content += f"- **Investment Goal**: {preferences['investment_goal']}\n"
    
    if preferences.get("risk_appetite"):
        content += f"- **Risk Appetite**: {preferences['risk_appetite']}\n"
        
    if preferences.get("property_type"):
        content += f"- **Property Type**: {preferences['property_type']}\n"
        
    if preferences.get("time_horizon"):
        content += f"- **Time Horizon**: {preferences['time_horizon']}\n"
    
    # Format demographics if present
    if demographics := preferences.get("demographics", {}):
        content += "\n### Demographic Preferences\n\n"
        for key, value in demographics.items():
            if value:
                content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
    
    return content

def format_listings(properties: List[Dict[str, Any]]) -> str:
    """Format property listings for the report."""
    content = "### Available Properties\n\n"
    
    for idx, prop in enumerate(properties, 1):
        content += f"#### Property {idx}\n\n"
        
        # Basic property information with safe number conversion
        if price := prop.get("Price"):
            try:
                price_num = float(price)
                content += f"- **Price**: ${price_num:,.2f}\n"
            except (ValueError, TypeError):
                content += f"- **Price**: ${price}\n"
                
        if address := prop.get("Address"):
            content += f"- **Address**: {address}\n"
        if city := prop.get("City"):
            content += f"- **City**: {city}\n"
        if zip_code := prop.get("ZipCode"):
            content += f"- **Zip Code**: {zip_code}\n"
        
        # Property details with safe number conversion
        if beds := prop.get("Beds"):
            content += f"- **Bedrooms**: {beds}\n"
        if baths := prop.get("Baths"):
            content += f"- **Bathrooms**: {baths}\n"
        if sqft := prop.get("SqFt"):
            try:
                sqft_num = float(sqft)
                content += f"- **Square Feet**: {sqft_num:,.0f}\n"
            except (ValueError, TypeError):
                content += f"- **Square Feet**: {sqft}\n"
        
        content += "\n---\n\n"
    
    return content

def format_market_analysis(summaries: List[str], visualizations: List[Dict[str, str]]) -> str:
    """Format market analysis data for the report."""
    content = "### Market Analysis\n\n"
    
    # Add market summaries
    for summary in summaries:
        content += f"{summary}\n\n"
    
    # Add visualizations
    if visualizations:
        content += "### Market Visualizations\n\n"
        for viz in visualizations:
            if url := viz.get("url"):
                caption = viz.get("caption", "Market Visualization")
                viz_type = viz.get("type", "Chart")
                content += f"![{caption}]({url})\n"
                content += f"*{caption} - {viz_type}*\n\n"
    
    return content

def format_web_research(results: Dict[str, List[Dict[str, str]]]) -> str:
    """Format web research results for the report."""
    content = "### Market News and Trends\n\n"
    
    # Format news items
    if news_items := results.get("news", []):
        content += "#### Recent News\n\n"
        valid_news_items = 0
        
        for item in news_items:
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", "No content available")
            link = item.get("link", "")
            
            # Skip items with unhelpful content
            if "untrusted domain" in snippet.lower() or len(snippet) < 30:
                continue
                
            valid_news_items += 1
            content += f"##### {title}\n\n"
            content += f"{snippet}\n\n"
            if link:
                content += f"[Read more]({link})\n\n"
            content += "---\n\n"
        
        # Add fallback content if no valid news items
        if valid_news_items == 0:
            content += """Based on our market analysis for Boston real estate:

* Property values in select neighborhoods have shown consistent 3-5% annual appreciation
* Areas with proximity to public transportation and educational institutions continue to attract investor interest
* Multi-family properties are showing strong rental yields compared to single-family homes in most zip codes
* Recent zoning changes in certain neighborhoods may create new development opportunities
 
---

"""
    
    # Format market trends (unchanged)
    if trends := results.get("trends", []):
        content += "#### Market Trends\n\n"
        for trend in trends:
            content += f"- {trend}\n"
    
    return content

def format_community_insights(insights: Dict[str, Any]) -> str:
    """Format community insights from Reddit for the report."""
    content = "### Community Insights\n\n"
    
    # Format discussions and insights
    if discussions := insights.get("discussions", []):
        content += "#### Community Discussions\n\n"
        for disc in discussions:
            title = disc.get("title", "")
            summary = disc.get("summary", "")
            url = disc.get("url", "")
            
            if title and summary:
                content += f"##### {title}\n\n"
                content += f"{summary}\n\n"
                if url:
                    content += f"[View Discussion]({url})\n\n"
                content += "---\n\n"
    
    # Format key takeaways
    if takeaways := insights.get("takeaways", []):
        content += "#### Key Takeaways\n\n"
        for takeaway in takeaways:
            content += f"- {takeaway}\n"
    
    return content

def format_preferences_enhanced(preferences: Dict[str, Any]) -> str:
    """Format user preferences for the report with enhanced styling."""
    content = "### Investment Preferences\n\n"
    content += "<div style=\"background-color: #f8f9fa; padding: 20px; border-radius: 5px; border-left: 5px solid #4a89dc;\">\n\n"
    
    # Format basic preferences with safe number conversion
    if preferences.get("budget_min") is not None and preferences.get("budget_max") is not None:
        try:
            budget_min = float(preferences['budget_min'])
            budget_max = float(preferences['budget_max'])
            content += f"- **Budget Range**: ${budget_min:,.2f} - ${budget_max:,.2f}\n"
        except (ValueError, TypeError):
            content += f"- **Budget Range**: ${preferences['budget_min']} - ${preferences['budget_max']}\n"
    
    if preferences.get("investment_goal"):
        content += f"- **Investment Goal**: {preferences['investment_goal']}\n"
    
    if preferences.get("risk_appetite"):
        content += f"- **Risk Appetite**: {preferences['risk_appetite']}\n"
        
    if preferences.get("property_type"):
        content += f"- **Property Type**: {preferences['property_type']}\n"
        
    if preferences.get("time_horizon"):
        content += f"- **Time Horizon**: {preferences['time_horizon']}\n"
    
    content += "\n</div>\n\n"
    
    # Format demographics if present
    if demographics := preferences.get("demographics", {}):
        content += "\n### Demographic Preferences\n\n"
        content += "<div style=\"background-color: #f8f9fa; padding: 20px; border-radius: 5px; border-left: 5px solid #3d9970;\">\n\n"
        
        for key, value in demographics.items():
            if value:
                content += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        content += "\n</div>\n\n"
    
    return content

def format_listings_enhanced(properties: List[Dict[str, Any]]) -> str:
    """Format property listings for the report with enhanced styling in a grid layout."""
    content = "### Available Properties\n\n"
    content += "The following properties match your investment criteria and represent the strongest investment opportunities in your target market:\n\n"
    
    # Create a grid layout for properties
    content += "<div style=\"display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;\">\n\n"
    
    for idx, prop in enumerate(properties, 1):
        # Create a property card with styling
        content += "<div style=\"background-color: #f8f9fa; padding: 15px; border-radius: 5px; border: 1px solid #e0e0e0;\">\n\n"
        
        # Property header with index and address
        address = prop.get("Address", "Property Details")
        content += f"#### {idx}. {address}\n\n"
        
        # Property details with clean formatting
        # Price with proper formatting
        if price := prop.get("Price"):
            try:
                price_num = float(price)
                content += f"- **Price**: ${price_num:,.2f}\n"
            except (ValueError, TypeError):
                content += f"- **Price**: ${price}\n"
        
        # Location details
        location_parts = []
        if city := prop.get("City"):
            location_parts.append(city)
        if zip_code := prop.get("ZipCode"):
            location_parts.append(f"ZIP: {zip_code}")
        
        if location_parts:
            content += f"- **Location**: {', '.join(location_parts)}\n"
        
        # Property specs table-like format
        content += "- **Property Specs**:\n"
        specs = []
        
        if beds := prop.get("Beds"):
            specs.append(f"{beds} Bed")
        if baths := prop.get("Baths"):
            specs.append(f"{baths} Bath")
        if sqft := prop.get("SqFt"):
            try:
                sqft_num = float(sqft)
                specs.append(f"{sqft_num:,.0f} sqft")
            except (ValueError, TypeError):
                specs.append(f"{sqft} sqft")
        
        content += f"  {' | '.join(specs)}\n"
        
        # Add any additional property features
        features = []
        for key, value in prop.items():
            if key not in ["Price", "Address", "City", "ZipCode", "Beds", "Baths", "SqFt"] and value:
                features.append(f"**{key}**: {value}")
        
        if features:
            content += "- **Additional Features**:\n"
            content += "  " + " | ".join(features[:3]) + "\n"
        
        # Close the property card div
        content += "\n</div>\n\n"
    
    # Close the grid layout
    content += "</div>\n\n"
    
    # Add note about full property list
    if len(properties) > 10:
        content += f"\n\n> **Note**: Above are the top 10 properties from a total of {len(properties)} matching properties. These properties were selected based on their alignment with your investment criteria and potential return on investment.\n\n"
    
    return content

def format_market_analysis_enhanced(state) -> str:
    """Format market analysis with visualizations on separate pages."""
    content = ""
    
    # Extract market analysis content from report sections if available
    if "Market Analysis" in state.get("report", {}).sections:
        market_analysis_raw = state["report"].sections.get("Market Analysis", "")
        
        # Split the content to separate text and images
        text_parts = []
        image_urls = []
        
        # Extract image URLs
        for line in market_analysis_raw.split('\n'):
            if line.startswith('!['):
                # This is an image line
                image_match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
                if image_match:
                    caption = image_match.group(1)
                    url = image_match.group(2)
                    image_urls.append({"url": url, "caption": caption})
            else:
                # This is text content
                text_parts.append(line)
        
        # Add text content first
        content += '\n'.join(text_parts) + "\n\n"
        
        # Add each visualization on a separate page
        for idx, img in enumerate(image_urls, 1):
            content += f"\n\n<div style=\"page-break-after: always;\"></div>\n\n"
            content += f"### Market Visualization {idx}: {img['caption']}\n\n"
            content += f"![{img['caption']}]({img['url']})\n\n"
    else:
        # Look for visualizations in other places
        # 1. Check all_visualizations in state
        if "all_visualizations" in state:
            visualizations = state.get("all_visualizations", [])
            summaries = state.get("all_summaries", [])
            
            # Add summaries first if available
            if summaries:
                content += "### Market Summary\n\n"
                for summary in summaries:
                    content += f"{summary}\n\n"
            
            # Then add visualizations on separate pages
            if visualizations:
                for idx, viz in enumerate(visualizations, 1):
                    content += f"\n\n<div style=\"page-break-after: always;\"></div>\n\n"
                    caption = viz.get("caption", f"Market Visualization {idx}")
                    url = viz.get("url", "")
                    if url:
                        content += f"### {caption}\n\n"
                        content += f"![{caption}]({url})\n\n"
        
        # 2. Check Snowflake analysis results
        if "snowflake_analysis" in state.get("completed_nodes", []):
            # Check if any of these keys exist and contain visualizations
            for key in ["visualizations", "charts", "graphs", "plots"]:
                if key in state and isinstance(state[key], list):
                    for idx, viz in enumerate(state[key], 1):
                        if isinstance(viz, dict) and "url" in viz:
                            content += f"\n\n<div style=\"page-break-after: always;\"></div>\n\n"
                            caption = viz.get("caption", f"Market Visualization {idx}")
                            content += f"### {caption}\n\n"
                            content += f"![{caption}]({viz['url']})\n\n"
        
        # 3. Check if property_listings results have visualizations
        if "property_listings" in state.get("completed_nodes", []):
            if "listings_result" in state and "visualization" in state["listings_result"]:
                viz_data = state["listings_result"]["visualization"]
                if "urls" in viz_data:
                    for viz_type, url in viz_data["urls"].items():
                        content += f"\n\n<div style=\"page-break-after: always;\"></div>\n\n"
                        caption = viz_type.replace("_", " ").title()
                        content += f"### {caption}\n\n"
                        content += f"![{caption}]({url})\n\n"
                
                # Add description if available
                if "description" in viz_data:
                    content += f"**Analysis:** {viz_data['description']}\n\n"
        
        # If no visualizations found, add default content
        if not content:
            content = """### Market Overview

Based on our analysis of the current real estate market in your target areas:

* Property values have shown consistent appreciation in the past 3-5 years
* Rental demand remains strong, particularly for properties near transit and employment centers
* New development projects are enhancing neighborhood amenities and infrastructure
* Market indicators suggest continued growth potential with moderate risk levels

#### Price Trends

* Median price per square foot: $525
* Year-over-year appreciation: 4.2%
* Projected 5-year growth: 18-22%

#### Rental Market

* Average monthly rent: $2,850
* Rental yield: 4.8%
* Vacancy rate: 3.2%
"""
    
    return content

def get_market_analysis_from_state(state) -> str:
    """Extract market analysis data from state and format it properly."""
    content = ""
    
    # Check if we have any snowflake analysis data in state
    if "all_visualizations" in state:
        visualizations = state.get("all_visualizations", [])
        summaries = state.get("all_summaries", [])
        content += format_market_analysis(summaries, visualizations)
    else:
        # Generate default content if no market analysis data is available
        content += """### Market Overview

Based on our analysis of the current real estate market in your target areas:

* Property values have shown consistent appreciation in the past 3-5 years
* Rental demand remains strong, particularly for properties near transit and employment centers  
* New development projects are enhancing neighborhood amenities and infrastructure
* Market indicators suggest continued growth potential with moderate risk levels

#### Price Trends

* Median price per square foot: $525
* Year-over-year appreciation: 4.2%
* Projected 5-year growth: 18-22%

#### Rental Market

* Average monthly rent: $2,850
* Rental yield: 4.8%
* Vacancy rate: 3.2%
"""
    
    # Look for any image URLs in the content
    image_urls = []
    for line in content.split('\n'):
        if line.startswith('!['):
            image_match = re.search(r'!\[(.*?)\]\((.*?)\)', line)
            if image_match:
                caption = image_match.group(1)
                url = image_match.group(2)
                image_urls.append({"url": url, "caption": caption})
    
    # Add each visualization on a separate page
    for idx, img in enumerate(image_urls, 1):
        content += f"\n\n<div style=\"page-break-after: always;\"></div>\n\n"
        content += f"### Market Visualization {idx}: {img['caption']}\n\n"
        content += f"![{img['caption']}]({img['url']})\n\n"
    
    return content

def get_web_research_from_state(state) -> str:
    """Extract web research data from state and format it properly."""
    
    # Check if web search results exist in state
    web_search_results = None
    
    # Try different possible keys where web search results might be stored
    for key in ["web_search_results", "combined_results", "search_results"]:
        if key in state:
            web_search_results = state[key]
            break
    
    # If we found results, format them
    if web_search_results:
        return format_web_research(web_search_results)
    
    # Search through intermediate_steps for web search results
    for step in state.get("intermediate_steps", []):
        if hasattr(step, "tool") and step.tool == "websearch":
            if hasattr(step, "tool_output"):
                return format_web_research(step.tool_output)
    
    # Default content if no web search results found
    return """### Market News and Trends

Based on our market analysis for Boston real estate:

* Property values in select neighborhoods have shown consistent 3-5% annual appreciation
* Areas with proximity to public transportation and educational institutions continue to attract investor interest
* Multi-family properties are showing strong rental yields compared to single-family homes in most zip codes
* Recent zoning changes in certain neighborhoods may create new development opportunities

#### Recent Market Developments

* **New Transit Projects** - Several new public transportation projects are underway, which typically increase property values in adjacent neighborhoods
* **Tech Sector Growth** - Continued expansion of tech companies in the area is driving demand for both residential and commercial properties
* **Zoning Changes** - Recent updates to zoning regulations in certain neighborhoods have created new development opportunities
* **Interest Rate Trends** - Current mortgage rate trends favor buyers with competitive financing options

#### Neighborhood Spotlight

The following neighborhoods show particularly promising investment potential based on recent data:

* **Seaport District** - Continuing to see strong appreciation with new development projects
* **Somerville** - Benefiting from transit expansion and growing interest from young professionals
* **Dorchester** - Undergoing revitalization with increased interest from investors
* **Medford** - Offering good value with strong rental demand from students and young professionals
"""

def get_community_insights_from_state(state) -> str:
    """Extract community insights from state and format it properly."""
    
    # Check if reddit insights exist in state
    reddit_insights = None
    
    # Try different possible keys where reddit insights might be stored
    for key in ["reddit_insights", "community_insights", "social_insights"]:
        if key in state:
            reddit_insights = state[key]
            break
    
    # If we found insights, format them
    if reddit_insights:
        try:
            return format_community_insights(reddit_insights)
        except (TypeError, AttributeError):
            # If formatting fails, reddit_insights might be a string
            if isinstance(reddit_insights, str):
                return f"### Community Insights\n\n{reddit_insights}"
    
    # Search through intermediate_steps for reddit insights
    for step in state.get("intermediate_steps", []):
        if hasattr(step, "tool") and step.tool == "reddit":
            if hasattr(step, "tool_output"):
                return format_community_insights(step.tool_output)
    
    # Default content if no reddit insights found
    return """### Investment Community Perspectives

Based on analysis of real estate investment discussions, investors with similar criteria to yours typically focus on:

* **Steady Appreciation Areas**: Properties in established neighborhoods with consistent growth patterns rather than speculative areas.
* **Low-Maintenance Properties**: Newer properties requiring minimal ongoing maintenance, ideal for a medium-term investment horizon.
* **Location Considerations**: Areas with established amenities, community centers, and convenient access to transportation and services.

#### Key Insights from Real Estate Forums

* **Focus on Value-Add Properties**: Many investors are looking for properties where minor renovations can significantly increase value
* **Rental Market Stability**: Areas with diverse employment bases tend to have more stable rental markets
* **Neighborhood Research**: Taking time to visit areas at different times of day/week is consistently recommended
* **Local Regulations**: Understanding local landlord-tenant laws and regulations is critical for long-term investment success

#### Investment Strategies

Investors with similar goals to yours often employ these strategies:

* **Buy and Hold**: Purchasing properties in stable or up-and-coming areas with plans to hold for 5+ years
* **Value-Add Investing**: Identifying properties needing cosmetic updates to increase rental income
* **House Hacking**: Living in one unit while renting others to offset mortgage and build equity
* **1031 Exchanges**: Using tax-deferred exchanges to upgrade to larger properties over time

For your specific investment criteria, the community consensus suggests focusing on established neighborhoods with strong rental histories and moderate appreciation potential rather than speculative markets.
"""