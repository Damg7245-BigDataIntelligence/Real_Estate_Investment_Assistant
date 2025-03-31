import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure the page
st.set_page_config(
    page_title="Boston Real Estate Investment Advisor",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
    }
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .stMetric {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stExpander {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 20px;
        margin: 0 5px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    .stMarkdown h1 {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stMarkdown h2 {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: bold;
    }
    .stMarkdown h3 {
        color: #34495e;
        font-size: 1.4rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "preferences": {
            "budget_min": None,
            "budget_max": None,
            "investment_goal": None,
            "risk_appetite": None,
            "property_type": None,
            "time_horizon": None,
            "demographics": {},
            "preferences": []
        },
        "is_complete": False,
        "current_step": "initial"
    }

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling"""
    with st.chat_message(role):
        st.markdown(content)

def get_ai_response(message: str) -> Dict:
    """Get response from the backend API"""
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                "http://localhost:8000/api/chat",
                json={
                    "message": message,
                    "state": st.session_state.state
                }
            )
            response.raise_for_status()
            return response.json()
    except Exception as e:
        st.error(f"Error communicating with the backend: {str(e)}")
        return None

def display_preference_progress():
    """Display progress of collected preferences"""
    preferences = st.session_state.state["preferences"]
    
    # Define required preferences with descriptions
    required_preferences = {
        "budget_min": "Minimum budget for investment",
        "budget_max": "Maximum budget for investment",
        "investment_goal": "Primary goal (e.g., rental income, appreciation)",
        "risk_appetite": "Risk tolerance (low, medium, high)",
        "property_type": "Type of property (e.g., single-family, condo)",
        "time_horizon": "Investment time period",
        "demographics": "Demographic preferences (if any)"
    }
    
    # Calculate progress (only count fields that are in required_preferences)
    total_fields = len(required_preferences)
    filled_fields = sum(
        1 for field in required_preferences 
        if preferences.get(field) is not None and preferences.get(field) != {} and preferences.get(field) != []
    )
    progress = min(filled_fields / total_fields, 1.0)  # Ensure progress doesn't exceed 1.0
    
    # Display progress bar
    st.progress(progress, text=f"Preferences collected: {filled_fields}/{total_fields}")
    
    # Display required preferences guide
    st.markdown("### 📋 Required Preferences")
    for field, description in required_preferences.items():
        status = "✅" if (preferences[field] is not None and preferences[field] != {} and preferences[field] != []) else "❌"
        st.markdown(f"{status} **{field.replace('_', ' ').title()}**: {description}")
    
    # Display collected preferences
    if filled_fields > 0:
        st.markdown("### 🎯 Your Preferences")
        col1, col2 = st.columns(2)
        with col1:
            if preferences["budget_min"] and preferences["budget_max"]:
                st.metric(
                    "Budget Range",
                    f"${preferences['budget_min']:,.0f} - ${preferences['budget_max']:,.0f}",
                    help="Your investment budget range"
                )
            if preferences["investment_goal"]:
                st.metric(
                    "Investment Goal",
                    preferences["investment_goal"],
                    help="Your primary investment objective"
                )
            if preferences["risk_appetite"]:
                st.metric(
                    "Risk Appetite",
                    preferences["risk_appetite"],
                    help="Your risk tolerance level"
                )
        with col2:
            if preferences["property_type"]:
                st.metric(
                    "Property Type",
                    preferences["property_type"],
                    help="Type of property you're interested in"
                )
            if preferences["time_horizon"]:
                st.metric(
                    "Time Horizon",
                    preferences["time_horizon"],
                    help="Your investment time period"
                )
            if preferences["demographics"]:
                st.metric(
                    "Demographic Preferences",
                    "Set",
                    help="Your demographic preferences for the area"
                )

def main():
    # Sidebar
    with st.sidebar:
        st.title("🏠 Real Estate Advisor")
        st.markdown("""
        ### About
        This AI-powered advisor helps you find the best neighborhoods in Boston based on your investment preferences.
        
        ### How it works
        1. Share your investment preferences
        2. AI analyzes your requirements
        3. Get personalized neighborhood recommendations
        4. View detailed analytics and insights
        
        ### 💡 Tips
        - Be specific about your preferences
        - Consider both short-term and long-term goals
        - Think about your risk tolerance
        - Consider demographic factors if important
        """)
        
        # Display progress and preferences guide
        display_preference_progress()
        
        # Add a reset button
        if st.button("🔄 Reset Preferences", type="secondary"):
            st.session_state.state = {
                "messages": [],
                "preferences": {
                    "budget_min": None,
                    "budget_max": None,
                    "investment_goal": None,
                    "risk_appetite": None,
                    "property_type": None,
                    "time_horizon": None,
                    "demographics": {},
                    "preferences": []
                },
                "is_complete": False,
                "current_step": "initial"
            }
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    st.title("Boston Real Estate Investment Advisor")
    st.markdown("""
        Welcome! I'm your AI investment advisor for Boston real estate. 
        I'll help you find the best neighborhoods based on your preferences.
        
        ### Getting Started
        I'll ask you about your investment preferences one by one. This will help me find the perfect neighborhoods for you.
        You can see your progress and required information in the sidebar.
        
        Let's begin! Tell me about your investment preferences...
    """)
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("Tell me about your investment preferences..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Get AI response
        response = get_ai_response(prompt)
        if response:
            # Update session state
            st.session_state.state = response["state"]
            
            # Add AI response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["message"]})
            display_chat_message("assistant", response["message"])
            
if __name__ == "__main__":
    main() 