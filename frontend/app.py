import streamlit as st
import requests
import time
import re
from typing import Dict
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Boston Real Estate Investment Advisor",
    page_icon="🏠",
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
    .report-container {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 20px 0;
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
if "research_task_id" not in st.session_state:
    st.session_state.research_task_id = None
if "research_report" not in st.session_state:
    st.session_state.research_report = None
if "view_mode" not in st.session_state:
    st.session_state.view_mode = "chat"  # Options: "chat", "report"

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling"""
    with st.chat_message(role):
        st.markdown(content)

def get_ai_response(message: str) -> Dict:
    """Get response from the backend API"""
    try:
        with st.spinner("Thinking..."):
            response = requests.post(
                "https://8703-71-192-242-59.ngrok-free.app/api/chat",
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

def check_research_status():
    """Check the status of the research task"""
    if not st.session_state.research_task_id:
        return False
    
    try:
        response = requests.get(
            f"https://8703-71-192-242-59.ngrok-free.app/api/research/{st.session_state.research_task_id}",
            timeout=10  # Add timeout to avoid hanging
        )
        
        # Debug the response
        print(f"API Response Code: {response.status_code}")
        if response.status_code != 200:
            st.warning(f"Error checking research status: HTTP {response.status_code}")
            return False
        
        result = response.json()
        print(f"Research status: {result.get('status')}")
        
        if result["status"] == "completed":
            # Get report content
            if "report" in result and result["report"]:
                report_length = len(result["report"])
                print(f"Received report with length: {report_length}")
                
                if report_length > 10:  # Make sure we have actual content
                    st.session_state.research_report = result["report"]
                    st.session_state.view_mode = "report"
                    
                    # Show success notification
                    st.success("Report generated successfully!")
                    st.balloons()  # Add a visual indicator
                    return True
                else:
                    print("Report is too short, may be empty")
                    # Try one more time with a direct request
                    try:
                        fresh_response = requests.get(
                            f"https://8703-71-192-242-59.ngrok-free.app/api/research/{st.session_state.research_task_id}?refresh=true",
                            timeout=15
                        )
                        if fresh_response.status_code == 200:
                            fresh_result = fresh_response.json()
                            if fresh_result.get("report") and len(fresh_result["report"]) > 10:
                                st.session_state.research_report = fresh_result["report"]
                                st.session_state.view_mode = "report"
                                st.success("Report generated successfully!")
                                st.balloons()
                                return True
                    except Exception as refresh_error:
                        print(f"Error in refresh attempt: {refresh_error}")
            else:
                print("No report field in response")
        
        return False
    except Exception as e:
        print(f"Error in check_research_status: {e}")
        return False

def process_markdown_for_images(markdown_text):
    """Process markdown text to handle images in reports"""
    if not markdown_text:
        return markdown_text
    
    # Find all image URLs in the markdown
    image_pattern = r"!\[(.*?)\]\((.*?)\)"
    
    def replace_image(match):
        alt_text = match.group(1)
        url = match.group(2)
        
        if url.startswith(("http://", "https://", "file://")):
            return f'<img src="{url}" alt="{alt_text}" style="max-width:100%; border-radius:5px; margin:10px 0;">'
        else:
            return match.group(0)  # Keep the original if not a full URL
    
    # Replace image markdown with HTML for better display
    processed_text = re.sub(image_pattern, replace_image, markdown_text)
    return processed_text

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

def display_report_view():
    """Display the comprehensive research report"""
    if not st.session_state.research_report:
        st.warning("No comprehensive report available yet.")
        return
    
    # Add download button and title in a single row
    col1, col2 = st.columns([5, 1])
    with col2:
        # Get the current date for the filename
        current_date = datetime.now().strftime("%Y%m%d")
        filename = f"real_estate_analysis_{current_date}.md"
        
        # Create download button
        st.download_button(
            label="📥 Download Report",
            data=st.session_state.research_report,
            file_name=filename,
            mime="text/markdown",
            key="download_report"
        )
    
    # Process markdown to handle images and display the report
    processed_report = process_markdown_for_images(st.session_state.research_report)
    
    # Display the report in a container
    st.markdown("<div class='report-container'>", unsafe_allow_html=True)
    st.markdown(processed_report, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Add button to return to chat
    if st.button("Return to Chat", key="return_to_chat"):
        st.session_state.view_mode = "chat"
        st.rerun()

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
        4. Request a comprehensive investment analysis
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
            st.session_state.research_task_id = None
            st.session_state.research_report = None
            st.session_state.view_mode = "chat"
            st.rerun()
        
        # Add view toggle buttons
        st.markdown("### 👀 View Mode")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💬 Chat", key="view_chat", disabled=st.session_state.view_mode == "chat"):
                st.session_state.view_mode = "chat"
                st.rerun()
        with col2:
            if st.button("📊 Report", key="view_report", 
                       disabled=st.session_state.view_mode == "report" or not st.session_state.research_report):
                st.session_state.view_mode = "report"
                st.rerun()
        
        # Add report status indicator if we have a research task
        if st.session_state.research_task_id:
            st.markdown("### 📊 Report Status")
            if st.session_state.research_report:
                st.success("✅ Report Ready!")
                # Add a button to view the report
                if st.button("📄 View Report"):
                    st.session_state.view_mode = "report"
                    st.rerun()
            else:
                st.info("⏳ Generating report...")
                # Add a debug button to check research status
                if st.button("🔍 Debug Report", key="debug_report"):
                    try:
                        debug_response = requests.get("https://8703-71-192-242-59.ngrok-free.app/api/debug/research", timeout=10)
                        debug_data = debug_response.json()
                        st.expander("Debug Information", expanded=True).json(debug_data)
                        
                        # If we find a report in the debug data, try to fetch it
                        for task in debug_data.get("recent_tasks", []):
                            if task.get("task_id") == st.session_state.research_task_id and task.get("has_report"):
                                st.info("Report found in debug data! Fetching...")
                                # Fetch the report directly
                                fetch_response = requests.get(
                                    f"https://8703-71-192-242-59.ngrok-free.app/api/research/{st.session_state.research_task_id}?refresh=true",
                                    timeout=15
                                )
                                if fetch_response.status_code == 200:
                                    fetch_data = fetch_response.json()
                                    if fetch_data.get("report"):
                                        st.session_state.research_report = fetch_data["report"]
                                        st.success("Report retrieved successfully!")
                                        st.rerun()
                                break
                    except Exception as e:
                        st.error(f"Debug error: {str(e)}")
    
    # Check research status if a task is running
    if st.session_state.research_task_id and not st.session_state.research_report:
        with st.status("Generating your investment report...", expanded=True) as status:
            st.write("🔍 Analyzing real estate data for your preferences...")
            
            # Add a progress display
            progress_bar = st.progress(0)
            
            # Define more specific status messages that don't repeat in a pattern
            status_messages = [
                "Initializing investment analysis",
                "Processing your investment preferences",
                "Identifying potential neighborhoods",
                "Querying property listings database",
                "Analyzing historical price trends",
                "Evaluating neighborhood amenities",
                "Calculating potential ROI metrics",
                "Researching local market conditions",
                "Evaluating school district performance",
                "Analyzing crime statistics",
                "Investigating transportation access",
                "Evaluating property tax implications",
                "Analyzing rental market demand",
                "Evaluating local economic indicators",
                "Compiling neighborhood profiles",
                "Generating investment recommendations",
                "Creating visualization charts",
                "Finalizing property analysis",
                "Preparing executive summary",
                "Compiling comprehensive report"
            ]
            
            # Check status multiple times with increasing progress
            max_checks = 20
            for i in range(max_checks):
                # Update progress
                progress_bar.progress((i + 1) / max_checks)
                
                # Show a unique status message for each step
                status_msg = status_messages[i] if i < len(status_messages) else f"Processing data (step {i+1}/{max_checks})"
                st.write(f"Step {i+1}/{max_checks}: {status_msg}")
                
                # Try to get the report status
                if check_research_status():
                    status.update(label="✅ Report Ready!", state="complete")
                    # Switch to report view automatically
                    st.session_state.view_mode = "report"
                    st.rerun()
                    break
                
                # Wait before checking again
                time.sleep(5)
            
            # If report still not ready after all checks
            if not st.session_state.research_report:
                try:
                    # Make one final attempt with a longer timeout
                    final_response = requests.get(
                        f"https://8703-71-192-242-59.ngrok-free.app/api/research/{st.session_state.research_task_id}?final=true",
                        timeout=20
                    )
                    if final_response.status_code == 200:
                        final_result = final_response.json()
                        if final_result.get("report") and len(final_result["report"]) > 10:
                            st.session_state.research_report = final_result["report"]
                            st.session_state.view_mode = "report"
                            status.update(label="✅ Report Ready!", state="complete")
                            st.rerun()
                except Exception as final_error:
                    print(f"Error in final report fetch: {final_error}")
                
                st.warning("Report generation is taking longer than expected. Please check back in a few minutes.")
    
    # Main content
    if st.session_state.view_mode == "report":
        display_report_view()
    else:
        st.title("Boston Real Estate Investment Advisor")
        
        # Display guidance for new users
        if not st.session_state.messages:
            st.markdown("""
                Welcome! I'm your AI investment advisor for Boston real estate. 
                I'll help you find the best neighborhoods based on your preferences.
                
                ### Getting Started
                I'll ask you about your investment preferences one by one. This will help me find the perfect neighborhoods for you.
                You can see your progress and required information in the sidebar.
                
                Once we've collected all your preferences, you can request a comprehensive investment analysis report.
                
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
                
                # Check if a research task was started
                if "research_task_id" in response and response["research_task_id"]:
                    st.session_state.research_task_id = response["research_task_id"]
                
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["message"]})
                display_chat_message("assistant", response["message"])
                
                # Rerun if a research task was started to show the status
                if st.session_state.research_task_id and not st.session_state.research_report:
                    st.rerun()
            
if __name__ == "__main__":
    main() 