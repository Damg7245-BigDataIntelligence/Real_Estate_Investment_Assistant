from typing import Dict, List, Any
from serpapi import GoogleSearch
import os
from dotenv import load_dotenv
from datetime import datetime
import json
from backend.llm_response import generate_response_with_gemini 

# Load environment variables
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".env"))
load_dotenv(dotenv_path)
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
print(SERPAPI_API_KEY)

class WebSearchAgent:
    def __init__(self):
        self.api_key = SERPAPI_API_KEY
        
    def search_news(self, query: str, num_results: int = 5, boston_specific = "") -> List[Dict[str, Any]]:
        """
        Search for recent news articles about NVIDIA
        """
        print(f"Searching for news articles about {query}")
        try:
            if boston_specific:
                nvidia_query = f"BOSTON real-estate: {query}"
            else:
                nvidia_query = f"USA real-estate: {query}"
            search_params = {
                "api_key": self.api_key,
                "engine": "google",
                "q": nvidia_query,
                "num": num_results,
                "tbm": "nws",  # News results
                "tbs": "qdr:m",  # Last month's results
                "location": "United States"
            }
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            formatted_results = []
            if "news_results" in results:
                for item in results["news_results"]:
                    formatted_results.append({
                        "type": "news",
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": item.get("source", ""),
                        "date": item.get("date", ""),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in news search: {str(e)}")
            return []

    def search_trends(self, query: str, num_results: int = 5, boston_specific = "") -> List[Dict[str, Any]]:
        """
        Search for general trends and articles about NVIDIA
        """
        try:
            if boston_specific:
                nvidia_query = f"BOSTON real-estate: {query} trends analysis research"
            else:
                nvidia_query = f"USA real-estate: {query} trends analysis research"
            search_params = {
                "api_key": self.api_key,
                "engine": "google",
                "q": nvidia_query,
                "num": num_results,
                "tbs": "qdr:m"  # Last month's results
            }
            
            search = GoogleSearch(search_params)
            results = search.get_dict()
            
            formatted_results = []
            if "organic_results" in results:
                for item in results["organic_results"]:
                    formatted_results.append({
                        "type": "trend",
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": item.get("source", "website"),
                        "date": item.get("date", "Recent"),
                        "timestamp": datetime.now().isoformat()
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error in trends search: {str(e)}")
            return []

    def process_results(self, news_results: List[Dict[str, Any]], trend_results: List[Dict[str, Any]], boston_specific = "") -> str:
        """
        Process and format both news and trend results into a comprehensive summary
        """
        if not news_results and not trend_results:
            return "No relevant results found."
            
        if boston_specific:
            summary = "Boston Real-Estate Intelligence Report:\n\n"
        else:
            summary = "USA Real-Estate Intelligence Report:\n\n"
        
        # Process News Section
        if news_results:
            summary += "📰 Latest News:\n" + "="*50 + "\n"
            for i, result in enumerate(news_results, 1):
                summary += f"{i}. {result['title']}\n"
                summary += f"   📅 {result['date']} | 🔍 {result['source']}\n"
                summary += f"   {result['snippet']}\n"
                summary += f"   🔗 {result['link']}\n\n"
        
        # Process Trends Section
        if trend_results:
            summary += "\n📈 Market Trends & Analysis:\n" + "="*50 + "\n"
            for i, result in enumerate(trend_results, 1):
                summary += f"{i}. {result['title']}\n"
                summary += f"   💡 Key Points: {result['snippet']}\n"
                summary += f"   🔗 {result['link']}\n\n"
        
        # Add timestamp
        summary += f"\n🕒 Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return summary

    def synthesize_results(self, news_results: List[Dict], trend_results: List[Dict], boston_specific = "") -> str:
        """
        Create an analytical summary using Gemini based on news and trend snippets
        """
        # Prepare context from news and trends
        news_context = "\n".join([
            f"NEWS ARTICLE:\n"
            f"Title: {item['title']}\n"
            f"Date: {item['date']}\n"
            f"Source: {item['source']}\n"
            f"Summary: {item['snippet']}\n"
            for item in news_results
        ])

        trends_context = "\n".join([
            f"MARKET TREND:\n"
            f"Title: {item['title']}\n"
            f"Summary: {item['snippet']}\n"
            for item in trend_results
        ])

        if boston_specific:
            cntx="Boston Real-estate"
        else:
            cntx="USA Real-estate"
        context = f"""
        RECENT NEWS AND TRENDS ABOUT {cntx}:

        {news_context}

        MARKET TRENDS AND ANALYSIS:
        {trends_context}
        """

        # Use the new response_type parameter
        analysis = generate_response_with_gemini(
            query=f"Analyze {cntx} updates",
            context=context,
        )
        
        return analysis  

    def run(self, query: str, boston_specific="") -> Dict[str, Any]:
        """
        Modified run method to include synthesis
        """
        try:
            # Perform searches
            news_results = self.search_news(query,boston_specific)
            trend_results = self.search_trends(query,boston_specific)
            
            # Generate basic summary
            summary = self.process_results(news_results, trend_results,boston_specific)
            
            # Generate analytical insights
            insights = self.synthesize_results(news_results, trend_results, boston_specific)
            
            return {
                "status": "success",
                "summary": summary,
                "insights": insights,  # New synthesized analysis
                "raw_results": {
                    "news": news_results,
                    "trends": trend_results
                },
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "categories": {
                    "has_news": bool(news_results),
                    "has_trends": bool(trend_results)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "query": query,
                "timestamp": datetime.now().isoformat()
            } 
        
agent = WebSearchAgent()
response = agent.run(query="real estate market", boston_specific="")
print(response)
print("---------------------------------------------")
agent = WebSearchAgent()
response = agent.run(query="real estate market", boston_specific="Boston")
print(response)