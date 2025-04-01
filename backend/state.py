from typing import List, Dict, Any, TypedDict, Optional, Union
from pydantic import BaseModel, Field
import os
from datetime import datetime

class UserPreferences(BaseModel):
    budget_min: float | None = None
    budget_max: float | None = None
    investment_goal: str | None = None
    risk_appetite: str | None = None
    property_type: str | None = None
    time_horizon: str | None = None
    demographics: Dict[str, Any] = {}
    preferences: List[str] = []
    demographics_asked: bool = False

class QueryAgentResponse(BaseModel):
    zip_codes: List[str]
    demographic_matches: List[Dict[str, Any]]
    preferences_complete: bool
    next_question: str | None = None
    error: str | None = None 

class ConversationState(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    preferences: Optional[Dict[str, Any]] = None
    preferences_complete: bool = False
    is_complete: bool = False
    current_step: str = "initial"
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
    
    def get_messages_text(self) -> str:
        """Get conversation history as text"""
        return "\n".join([msg["content"] for msg in self.messages])
    
class AgentAction:
    """Action returned by agent."""
    def __init__(self, tool: str, tool_input: Dict, log: str = ""):
        self.tool = tool
        self.tool_input = tool_input
        self.log = log

class RealEstateReport(BaseModel):
    report_file: str | None = None
    sections: Dict[str, Any] = Field(default_factory=dict)

    def append_section(self, section_name: str, content: str):
        """Append content to the report file under a section"""
        if not self.report_file:
            raise ValueError("Report file path not set")
            
        with open(self.report_file, "a", encoding='utf-8') as f:
            f.write(f"\n## {section_name}\n\n")
            f.write(content)
            f.write("\n\n---\n\n")
            f.flush()

    def initialize_file(self, task_id: str) -> str:
        """Initialize report file if not exists"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = f"reports/{task_id}_{timestamp}.md"
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(self.report_file), exist_ok=True)
        
        # Write initial headers
        with open(self.report_file, "w", encoding='utf-8') as f:
            f.write("# Real Estate Investment Analysis Report\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*\n\n")
            f.write("## Table of Contents\n\n")
            f.write("1. [User Preferences](#user-preferences)\n")
            f.write("2. [Property Listings](#property-listings)\n")
            f.write("3. [Market Analysis](#market-analysis)\n")
            f.write("4. [Web Research](#web-research)\n")
            f.write("5. [Community Insights](#community-insights)\n")
            f.write("6. [Investment Recommendations](#investment-recommendations)\n\n")
            f.write("---\n\n")
            f.flush()
            
        return self.report_file

    def update_section(self, section_key: str, data: Any):
        """Store data in sections dictionary and optionally append to file"""
        self.sections[section_key] = data
        
        # If this is a simple piece of text, we could also append it
        if isinstance(data, str) and self.report_file:
            with open(self.report_file, "a", encoding='utf-8') as f:
                f.write(f"\n## {section_key}\n\n")
                f.write(data)
                f.write("\n\n---\n\n")
                f.flush()

class RealEstateState(TypedDict):
    """State for the Real Estate Investment Report Agent."""
    input: str  # User's query
    user_preferences: Dict[str, Any]  # User's investment preferences
    zip_codes: List[str]  # Zip codes for analysis
    cities: List[str]  # Cities for analysis
    properties: List[Dict[str, Any]]  # Property listings
    chat_history: List  # Conversation history
    intermediate_steps: List[AgentAction]  # Results from agent actions
    report: Optional[RealEstateReport]  # The final report container
    current_node: str  # Track the current node in the graph
    completed_nodes: List[str]  # Track which nodes have been completed
    error: Optional[str]  # Store any errors that occur
