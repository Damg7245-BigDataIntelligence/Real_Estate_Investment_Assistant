import google.generativeai as genai
from typing import Dict, Any
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-pro')

class LLMService:
    @staticmethod
    async def get_llm_response(prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Get response from LLM with optional system prompt
        
        Args:
            prompt (str): User prompt
            system_prompt (str, optional): System prompt to set context
            
        Returns:
            Dict[str, Any]: Response from LLM
        """
        try:
            # Combine system prompt and user prompt if both exist
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            # Generate response
            response = model.generate_content(full_prompt)
            
            # Handle different response types
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'parts'):
                response_text = response.parts[0].text
            else:
                response_text = str(response)
            
            # Clean up the response text
            response_text = response_text.strip()
            
            # If the response starts with ```json and ends with ```, extract the JSON part
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            return {
                "success": True,
                "response": response_text
            }
            
        except Exception as e:
            print(f"Error in get_llm_response: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def get_sql_query(prompt: str) -> Dict[str, Any]:
        """
        Get SQL query from LLM based on user preferences
        
        Args:
            prompt (str): User preferences and requirements
            
        Returns:
            Dict[str, Any]: SQL query and explanation
        """
        system_prompt = """You are an expert real estate investment advisor specializing in Boston area properties.
        Your task is to analyze user preferences and generate SQL queries to find matching neighborhoods.
        
        Available data columns in the SUFFOLK_REAL_ESTATE_DEMOGRAPHICS table:
        - ZIP_CODE: Area code
        - YEAR: Data year (2020-2023)
        - TOTAL_POPULATION: Total residents
        - WHITE_POP, BLACK_POP, ASIAN_POP: Racial demographics
        - MALE_POP, FEMALE_POP: Gender demographics
        - TOTAL_HOUSING_UNITS: Available housing
        - AVERAGE_HOUSE_VALUE: Median property value
        - INCOME_PER_HOUSEHOLD: Median household income
        - PERSONS_PER_HOUSEHOLD: Average household size
        
        Generate SQL queries that:
        1. Use DISTINCT to avoid duplicate results
        2. Match user's budget constraints
        3. Consider demographic preferences
        4. Factor in investment goals and risk appetite
        5. Look at population trends
        6. Return multiple matching neighborhoods for comparison
        
        Example SQL query format:
        SELECT DISTINCT
            ZIP_CODE,
            AVERAGE_HOUSE_VALUE,
            INCOME_PER_HOUSEHOLD,
            TOTAL_POPULATION,
            WHITE_POP, BLACK_POP, ASIAN_POP,
            TOTAL_HOUSING_UNITS
        FROM SUFFOLK_REAL_ESTATE_DEMOGRAPHICS
        WHERE YEAR = 2023
        AND AVERAGE_HOUSE_VALUE BETWEEN :budget_min AND :budget_max
        ORDER BY AVERAGE_HOUSE_VALUE ASC;
        
        Format your response as a JSON with:
        {
            "sql_query": "your SQL query here",
            "explanation": "explanation of why this query matches user preferences"
        }
        """
        response = await LLMService.get_llm_response(prompt, system_prompt)
        print("SQL query response:", response.get("response"))
        return response
    
    @staticmethod
    async def get_next_question(user_input: str, current_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get next question to ask user based on current preferences
        
        Args:
            user_input (str): User's current input
            current_preferences (Dict[str, Any]): Current preferences collected
            
        Returns:
            Dict[str, Any]: Next question and updated preferences
        """
        system_prompt = """You are a real estate investment advisor assistant.
        Your task is to collect all necessary information from the user through a conversation.
        
        Required information:
        1. Budget range (min and max)
        2. Investment goal (e.g., rental income, appreciation, etc.)
        3. Risk appetite (low, medium, high)
        4. Property type preference
        5. Time horizon
        6. Demographic preferences (if any)
        7. Additional preferences
        
        COMPLETION RULES:
        Set is_complete to true when ALL of these conditions are met:
        1. budget_min and budget_max are both set (not null)
        2. investment_goal is set (not null)
        3. risk_appetite is set (not null)
        4. property_type is set (not null)
        5. time_horizon is set (not null)
        6. User has been asked about demographic preferences
        7. User has indicated they have no more preferences to add
        
        IMPORTANT: When the user says "no" to additional preferences and all required fields are filled,
        you MUST set is_complete to true and stop asking for more preferences.
        
        Response Format:
        {
            "next_question": "your next question here",
            "preferences": {
                "budget_min": float or null,
                "budget_max": float or null,
                "investment_goal": str or null,
                "risk_appetite": str or null,
                "property_type": str or null,
                "time_horizon": str or null,
                "demographics": dict or {},
                "preferences": list or []
            },
            "is_complete": boolean
        }
        
        Check current preferences before asking for more:
        - Only ask about missing preferences
        - If all required fields are filled and user says "no" to more preferences, set is_complete to true
        - Don't ask about the same preference multiple times
        """
        
        prompt = f"""
        Current user input: {user_input}
        Current preferences: {current_preferences}
        
        Extract preferences from the user's input and determine the next question.
        Remember to return a valid JSON object with the exact structure specified.
        """
        
        try:
            response = await LLMService.get_llm_response(prompt, system_prompt)
            
            if not response["success"]:
                print(f"LLM response error: {response.get('error')}")
                return {
                    "success": False,
                    "error": "Failed to get LLM response"
                }
            
            print(f"Raw LLM response: {response['response']}")
            
            # Try to parse the response as JSON
            try:
                response_data = json.loads(response["response"])
                print(f"Parsed JSON response: {response_data}")
                return {
                    "success": True,
                    "response": json.dumps(response_data)
                }
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {str(e)}")
                print(f"Attempting to fix response: {response['response']}")
                
                # If JSON parsing fails, try to fix common issues
                fixed_response = response["response"].replace("'", '"')
                try:
                    response_data = json.loads(fixed_response)
                    print(f"Successfully fixed and parsed JSON: {response_data}")
                    return {
                        "success": True,
                        "response": json.dumps(response_data)
                    }
                except json.JSONDecodeError as e:
                    print(f"Failed to fix JSON: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Failed to parse LLM response as JSON: {str(e)}"
                    }
                    
        except Exception as e:
            print(f"Unexpected error in get_next_question: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            } 