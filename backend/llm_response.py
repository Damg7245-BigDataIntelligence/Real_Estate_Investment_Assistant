import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
def get_gemini_model():
    dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(dotenv_path)

    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

    genai.configure(api_key=GOOGLE_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-pro")

    return gemini_model


def reddit_select_titles(subreddit_name,title_list):
    gemini_model = get_gemini_model()
    prompt = f"""
        You're helping build a Boston real estate recommendation app. Review these Reddit post titles from r/{subreddit_name} and identify ones that might contain useful insights for property buyers/investors in Boston.

        Use your judgment to select titles that:
        - Discuss Boston real estate trends, neighborhoods, or investment strategies
        - Contain housing market information (pricing, rentals, buying/selling)
        - Include personal experiences relevant to Boston housing
        - Offer advice applicable to property decisions

        Include anything that will help the application find information on:
        1. Market Trends & Pricing Intelligence: Insights on Boston housing market conditions, pricing patterns, inventory changes, forecasts, or market comparisons.
        2. Neighborhood & Property Insights: Information about specific Boston areas (e.g., safety, amenities, schools, transportation) and community feedback.
        3. Investment Strategies & Financial Considerations: Details on ROI analysis, financing options, tax strategies, or investment approaches for Boston properties.
        4. Rental Market Dynamics: Information on Boston rental trends, vacancy patterns, tenant rights, landlord experiences, or rental yields.
        5. First-Time Buyer Guidance: Advice for new buyers in the Boston market, including common pitfalls and inspection insights.
        6. Property-Specific Considerations: Discussions about property types, features, or specific properties in the Boston area.

        Exclude titles that:
        - Are promotional content for specific listings or services.
        - Do not relate directly to Boston real estate.
        - Are general lifestyle posts unrelated to property investment or market trends.
        - Are too vague or generic to provide actionable insights.
        - Personal questions that are too vague

        Title list: {title_list}

        Return:
        Just the relevant titles separated by a newline (`\n`). Do not return anything else (no Python list format or explanations).
        """

    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def reddit_generate_report(context, query, input_info):
    # Create context for Gemini
    prompt = f"""
    You are a professional real estate assistant tasked with analyzing Boston real estate trends for the user.

    Below is the relevant property market data retrieved from a vector database:
    - **Context:** {context}

    Additionally, here is some user-specific information:
    - **User Information:** {input_info}

    ### Question:
    {query}

    ### Instructions:
    1. Use the provided context and user information to provide an accurate and concise analysis dont mention anything about user.
    2. Ensure that the report integrates relevant insights from additional sources, presents a structured response, and avoids repetition.
    3. Act as a real estate expert providing the user with a final report in a professional and actionable tone.
    4. The report should be well-structured and clear, summarizing key insights for the user, and should be written as if you are directly communicating with the user.
    5. Do not mention anything about the context or the process. The user should not be aware of the underlying context or data sources.
    7. Do not mention anaything about user, username, client name, date, (PDF Format - Suitable for Client Delivery) or anything related. 

    Provide the output in a concise and structured report, similar to what a real estate consultant would prepare for a client
    Answer in markdown format in 750-800 words. Begin the answer with a heading is ## format in markdown and heading ahould summarize query.
    Don't give a disclaimer at the end.
    """

    gemini_model = get_gemini_model()
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

def generate_response_with_gemini(query, context):
    prompt = f"""
    You are a financial analyst specializing in real estate investments.
    Analyze the following recent news and market trends to provide strategic insights for property recommendations.

    {context}

    Please provide a structured analysis with the following sections:

    1. KEY DEVELOPMENTS:
    - List the most significant recent events or policy changes in the real estate market.
    - Highlight their relevance for Boston’s market dynamics.

    2. MARKET IMPACT:
    - Evaluate potential effects on property values and investment opportunities in Boston.
    - Discuss broader national trends that could influence local markets.

    3. LOCAL MARKET INSIGHTS (Boston-specific):
    - Analyze Boston’s unique market conditions and real estate trends.
    - Identify any challenges or opportunities specific to the city.

    4. FUTURE OUTLOOK:
    - Provide forward-looking analysis on Boston’s real estate market.
    - Highlight potential opportunities, risks, and emerging trends.
    - Suggest key indicators or areas to monitor.

    Focus on factual analysis based on the provided information.
    Query: {query}
    """
    gemini_model = get_gemini_model()
    response = gemini_model.generate_content(prompt)
    return response.text.strip()

