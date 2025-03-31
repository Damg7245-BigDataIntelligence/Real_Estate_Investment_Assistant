from typing import Dict, Any, List
from pydantic import BaseModel

class UserPreferences(BaseModel):
    budget_min: float | None = None
    budget_max: float | None = None
    investment_goal: str | None = None
    risk_appetite: str | None = None
    property_type: str | None = None
    time_horizon: str | None = None
    demographics: Dict[str, Any] = {}
    preferences: List[str] = []

class ConversationState(BaseModel):
    messages: List[Dict[str, str]] = []
    preferences: UserPreferences = UserPreferences()
    is_complete: bool = False
    current_step: str = "initial"
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
    
    def update_preferences(self, new_preferences: Dict[str, Any]):
        """Update user preferences"""
        for key, value in new_preferences.items():
            if hasattr(self.preferences, key):
                setattr(self.preferences, key, value)
    
    def get_messages_text(self) -> str:
        """Get conversation history as text"""
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.messages]) 