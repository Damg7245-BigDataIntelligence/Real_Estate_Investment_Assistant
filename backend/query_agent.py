from typing import Dict, Any
from llm_service import LLMService
from backend.state import UserPreferences

class QueryAgent:
    def __init__(self):
        self.llm_service = LLMService()
    
    async def generate_query(self, preferences: UserPreferences) -> Dict[str, Any]:
        """
        Generate SQL query based on user preferences
        
        Args:
            preferences (UserPreferences): User's investment preferences
            
        Returns:
            Dict[str, Any]: SQL query and explanation
        """
        # Format preferences for LLM
        prompt = f"""
        Generate a SQL query to find matching neighborhoods based on these preferences:
        
        Budget: ${preferences.budget_min:,.2f} - ${preferences.budget_max:,.2f}
        Investment Goal: {preferences.investment_goal}
        Risk Appetite: {preferences.risk_appetite}
        Property Type: {preferences.property_type}
        Time Horizon: {preferences.time_horizon}
        
        Demographic Preferences:
        {preferences.demographics}
        
        Additional Preferences:
        {', '.join(preferences.preferences)}
        
        Return the most recent year's data (2023) and consider population trends.
        """
        print("Query agent prompt:", prompt)
        return await self.llm_service.get_sql_query(prompt) 