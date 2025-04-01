import os
import asyncio
from tavily import TavilyClient
from dotenv import load_dotenv
from urllib.parse import urlparse
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from llm_response import generate_response_with_gemini

load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

TRUSTED_DOMAINS = {
    "mass.gov", "cityofboston.gov", "census.gov", "bls.gov",
    "bostonplans.org", "urban.org", "fred.stlouisfed.org", "boston.com",
    "bizjournals.com", "masshousing.com", "bostonhousing.org"
}

class WebSearchAgent:
    def __init__(self):
        self.client = TavilyClient(TAVILY_API_KEY)

    async def analyze_zipcode(self, zipcode: str):
        try:
            zipcode = str(zipcode).replace(".0", "").zfill(5)

            queries = [
                f"Boston {zipcode} neighborhood development site:{' OR site:'.join(TRUSTED_DOMAINS)}",
                f"{zipcode} Boston economic indicators housing employment site:{' OR site:'.join(TRUSTED_DOMAINS)}",
                f"{zipcode} Boston population trends and demographics site:{' OR site:'.join(TRUSTED_DOMAINS)}",
                f"{zipcode} Boston infrastructure transportation projects site:{' OR site:'.join(TRUSTED_DOMAINS)}",
                f"{zipcode} real estate market outlook risks opportunities site:{' OR site:'.join(TRUSTED_DOMAINS)}"
            ]

            all_snippets = []

            for query in queries:
                try:
                    results = self.client.search(
                        query=query,
                        search_depth="advanced",
                        max_results=3,
                        include_domains=list(TRUSTED_DOMAINS),
                        exclude_domains=["realtor.com", "zillow.com", "redfin.com"]
                    )

                    for r in results.get("results", []):
                        domain = urlparse(r["url"]).netloc.replace("www.", "")
                        if domain in TRUSTED_DOMAINS and r.get("description"):
                            all_snippets.append(f"{r['title']}\n{r['description']}\nSource: {r['url']}")

                except Exception as e:
                    print(f"Error fetching Tavily results: {e}")
                    continue

            # Prepare final prompt for Gemini
            combined_info = "\n\n".join(all_snippets[:10]) or "No data found."

            prompt = f"""
            You are a real estate market analyst. Based on the information provided below for Boston ZIP code {zipcode},
            create a detailed investment report covering:

            - Recent or upcoming development projects
            - Local economic indicators (jobs, income, housing)
            - Population and demographic changes
            - Public infrastructure and transportation news
            - Risks and opportunities in the real estate market

            Rely on official, public, or local media data provided in the context.

            Context:
            {combined_info}
            """

            analysis = generate_response_with_gemini(prompt, context=combined_info)

            # Return a dictionary instead of just the analysis string
            return {
                "success": True,
                "zipcode": zipcode,
                "results": {
                    "development_projects": all_snippets[:3],
                    "market_trends": all_snippets[3:6]
                },
                "analysis": analysis
            }

        except Exception as e:
            print(f"Error analyzing zipcode: {e}")
            return {
                "success": False,
                "error": str(e),
                "zipcode": zipcode,
                "results": {
                    "development_projects": [],
                    "market_trends": []
                },
                "analysis": ""
            }

# --- Usage ---
# async def run():
#     agent = WebSearchAgent()
#     zip_code = "02127"  # example ZIP
#     result = await agent.analyze_zipcode(zip_code)
#     print("\n====== INVESTMENT ANALYSIS ======\n")
#     print(result)

# if __name__ == "__main__":
#     asyncio.run(run())
