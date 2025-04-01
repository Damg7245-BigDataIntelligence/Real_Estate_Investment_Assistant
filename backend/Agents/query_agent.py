from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
from state import UserPreferences, QueryAgentResponse
import snowflake.connector
import asyncio
import re
import traceback
load_dotenv()

class QueryAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.preferences = UserPreferences()
        self.preferences_complete = False
        
        # Add Snowflake connection
        self.snowflake_conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema='SUFFOLK_REAL_ESTATE_SCHEMA'  # Hardcoded as specified
        )

        # Add valid locations
        self.valid_locations = {
            'BOSTON', 'DORCHESTER', 'REVERE', 'CHELSEA'
        }
        self.valid_demographics = {
            'asian_community': 'ASIAN_POP',
            'black_community': 'BLACK_POP',
            'white_community': 'WHITE_POP'
        }

    def process_input(self, user_input: str) -> QueryAgentResponse:
        """Process user input and return matching zip codes and demographics"""
        try:
            # First, update preferences based on user input
            updated_prefs = self._update_preferences(user_input)
            
            # Check if preferences are complete
            if self._are_preferences_complete():
                print("All preferences complete, generating recommendations")
                self.preferences_complete = True
                return self._generate_recommendations()
            
            # If preferences are not complete, return next question
            next_question = self._get_next_question()
            print("Next question:", next_question)
            
            return QueryAgentResponse(
                zip_codes=[],
                demographic_matches=[],
                preferences_complete=False,
                next_question=next_question
            )
            
        except Exception as e:
            print(f"Error in process_input: {e}")
            return QueryAgentResponse(
                zip_codes=[],
                demographic_matches=[],
                preferences_complete=False,
                error=str(e)
            )

    def _update_preferences(self, user_input: str) -> Dict[str, Any]:
        """Update preferences based on user input"""
        try:
            print("\nProcessing input for preferences:", user_input)
            input_lower = user_input.lower()
            
            # Budget parsing with improved patterns
            budget_patterns = [
                # Match "800k - 1M" or "800k-1M" or "800k to 1M"
                (r'(\d+(?:\.\d+)?)\s*[kK]\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*[mM][nN]?', 
                 lambda x, y: (float(x) * 1000, float(y) * 1000000)),
                
                # Match "1M - 2M" or "1Mn - 2Mn"
                (r'(\d+(?:\.\d+)?)\s*[mM][nN]?\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*[mM][nN]?',
                 lambda x, y: (float(x) * 1000000, float(y) * 1000000)),
                
                # Match "800k - 1000k"
                (r'(\d+(?:\.\d+)?)\s*[kK]\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*[kK]',
                 lambda x, y: (float(x) * 1000, float(y) * 1000)),
                
                # Match "800000 - 1000000"
                (r'(\d{6,})\s*(?:-|to)\s*(\d{6,})',
                 lambda x, y: (float(x), float(y)))
            ]
            
            budget_found = False
            for pattern, converter in budget_patterns:
                match = re.search(pattern, user_input)
                if match:
                    try:
                        min_val, max_val = converter(match.group(1), match.group(2))
                        self.preferences.budget_min = min_val
                        self.preferences.budget_max = max_val
                        budget_found = True
                        print(f"Found budget range: ${min_val:,.2f} - ${max_val:,.2f}")
                        break
                    except Exception as e:
                        print(f"Error converting budget values: {e}")
                        continue
            
            # If no pattern matched but input contains budget-related keywords, try simpler parsing
            if not budget_found and any(word in input_lower for word in ['budget', 'price', 'cost', '$', 'dollar']):
                # Find all numbers in the input
                numbers = re.findall(r'(\d+(?:\.\d+)?)\s*([kKmMnN])?', input_lower)
                if len(numbers) >= 2:
                    min_val = float(numbers[0][0])
                    max_val = float(numbers[1][0])
                    
                    # Apply multipliers
                    for val, unit in [numbers[0], numbers[1]]:
                        if unit.lower() == 'k':
                            val = float(val) * 1000
                        elif unit.lower() in ['m', 'mn']:
                            val = float(val) * 1000000
                    
                    self.preferences.budget_min = min_val
                    self.preferences.budget_max = max_val
                    print(f"Found budget range (simple parse): ${min_val:,.2f} - ${max_val:,.2f}")

            # Check for investment goal
            goal_keywords = {
                "rental": "rental_income",
                "rent": "rental_income",
                "appreciation": "appreciation",
                "value": "appreciation",
                "flip": "property_flipping"
            }
            
            for keyword, goal in goal_keywords.items():
                if keyword in input_lower:
                    self.preferences.investment_goal = goal
                    print(f"Found investment goal: {goal}")
                    break

            # Check for risk appetite
            risk_keywords = ["low", "medium", "high"]
            for risk in risk_keywords:
                if risk in input_lower:
                    self.preferences.risk_appetite = risk
                    print(f"Found risk appetite: {risk}")
                    break

            # Check for property type
            property_types = {
                "house": "house",
                "apartment": "apartment",
                "condo": "condo",
                "townhouse": "townhouse"
            }
            
            for keyword, prop_type in property_types.items():
                if keyword in input_lower:
                    self.preferences.property_type = prop_type
                    print(f"Found property type: {prop_type}")
                    break

            # Check for time horizon
            time_patterns = [
                (r'(\d+)\s*years?', 'years'),
                (r'(\d+)\s*months?', 'months')
            ]
            
            for pattern, unit in time_patterns:
                match = re.search(pattern, input_lower)
                if match:
                    duration = match.group(1)
                    self.preferences.time_horizon = f"{duration} {unit}"
                    print(f"Found time horizon: {duration} {unit}")
                    break

            # Handle demographic preferences
            if not self.preferences.demographics_asked:
                if 'no preferences' in input_lower:
                    self.preferences.demographics = {}
                    self.preferences.demographics_asked = True
                    print("No demographic preferences specified")
                else:
                    demographics = {}
                    for key in self.valid_demographics:
                        if key.split('_')[0] in input_lower:
                            demographics[key] = True
                            print(f"Found demographic preference: {key}")
                    if demographics:
                        self.preferences.demographics = demographics
                        self.preferences.demographics_asked = True

            print("\nFinal preferences after update:")
            print(self.preferences.model_dump_json(indent=2))
            
            return self.preferences.model_dump()

        except Exception as e:
            print(f"Error in _update_preferences: {e}")
            traceback.print_exc()
            return {}

    def _are_preferences_complete(self) -> bool:
        """Check if all required preferences are collected"""
        # Basic preferences check
        basic_complete = all([
            self.preferences.budget_min is not None,
            self.preferences.budget_max is not None,
            self.preferences.investment_goal is not None,
            self.preferences.risk_appetite is not None,
            self.preferences.property_type is not None,
            self.preferences.time_horizon is not None
        ])
        
        # If basic preferences aren't complete, return False
        if not basic_complete:
            return False
            
        # If demographics haven't been asked about, return False
        if not self.preferences.demographics_asked:
            return False
            
        # All preferences are complete
        return True

    def _get_next_question(self) -> str:
        """Generate next question based on missing preferences"""
        # Check which basic preferences are missing
        if self.preferences.budget_min is None or self.preferences.budget_max is None:
            return "What is your budget range for the investment?"
        if self.preferences.investment_goal is None:
            return "What is your investment goal (e.g., rental income, appreciation)?"
        if self.preferences.risk_appetite is None:
            return "What is your risk appetite (low, medium, high)?"
        if self.preferences.property_type is None:
            return "What type of property are you interested in (e.g., house, apartment)?"
        if self.preferences.time_horizon is None:
            return "What is your investment time horizon?"
        
        # If demographics haven't been asked about yet
        if not self.preferences.demographics_asked:
            return "Do you have any specific community preferences? We serve Asian, Black, and White communities. Please specify your preference or say 'no preferences'."
        
        # If we get here, all preferences are complete
        return None

    def _generate_recommendations(self) -> QueryAgentResponse:
        """Generate recommendations once preferences are complete"""
        try:
            # First try with all preferences
            sql_result = self._generate_sql_query(relaxation_level=0)
            print("First SQL Generation Result:", sql_result)
            
            query_data = self._parse_and_check_results(sql_result)
            if query_data and query_data.get("matching_zip_codes"):
                return QueryAgentResponse(
                    zip_codes=query_data["matching_zip_codes"],
                    demographic_matches=query_data["demographic_matches"],
                    preferences_complete=True
                )

            # If no results, try with relaxed constraints
            sql_result = self._generate_sql_query(relaxation_level=1)
            print("Second SQL Generation Result (Relaxed):", sql_result)
            
            query_data = self._parse_and_check_results(sql_result)
            if query_data and query_data.get("matching_zip_codes"):
                return QueryAgentResponse(
                    zip_codes=query_data["matching_zip_codes"],
                    demographic_matches=query_data["demographic_matches"],
                    preferences_complete=True
                )

            # If still no results, try with minimal constraints
            sql_result = self._generate_sql_query(relaxation_level=2)
            print("Third SQL Generation Result (Minimal):", sql_result)
            
            query_data = self._parse_and_check_results(sql_result)
            return QueryAgentResponse(
                zip_codes=query_data.get("matching_zip_codes", []),
                demographic_matches=query_data.get("demographic_matches", []),
                preferences_complete=True
            )

        except Exception as e:
            print(f"Error in generate_recommendations: {e}")
            return QueryAgentResponse(
                zip_codes=[],
                demographic_matches=[],
                preferences_complete=True,
                error=str(e)
            )

    def _parse_and_check_results(self, sql_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse SQL result and execute query on Snowflake"""
        if not sql_result["success"]:
            return None
        
        try:
            query_data = json.loads(sql_result["response"])
            
            # Extract the SQL query
            sql_query = query_data.get("sql_query")
            if not sql_query:
                return None
                
            # Execute the query on Snowflake
            snowflake_results = self._execute_snowflake_query(sql_query)
            
            if not snowflake_results["success"]:
                print(f"Snowflake query error: {snowflake_results['error']}")
                return None
                
            # Process the actual results from Snowflake
            matching_zip_codes = []
            demographic_matches = []
            
            for row in snowflake_results["data"]:
                zip_code = str(row["ZIP_CODE"])
                matching_zip_codes.append(zip_code)
                
                # Create demographics dictionary with all available demographic data
                demographics = {
                    "average_house_value": float(row["AVERAGE_HOUSE_VALUE"]),
                    "income_per_household": float(row["INCOME_PER_HOUSEHOLD"]),
                    "total_population": int(row["TOTAL_POPULATION"])
                }
                
                # Add demographic data if available
                for demo in ["ASIAN_POP", "BLACK_POP", "WHITE_POP"]:
                    if demo in row:
                        demographics[demo.lower()] = int(row[demo])
                
                demographic_matches.append({
                    "zip_code": zip_code,
                    "demographics": demographics
                })
            
            return {
                "matching_zip_codes": matching_zip_codes,
                "demographic_matches": demographic_matches,
                "user_preferences": self.preferences.model_dump()  # Include user preferences in response
            }
            
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            return None
        except Exception as e:
            print(f"Error in parse_and_check_results: {str(e)}")
            return None

    def _generate_sql_query(self, relaxation_level: int = 0) -> Dict[str, Any]:
        """Generate SQL query based on preferences with different relaxation levels"""
        
        # Base conditions that we always keep
        base_conditions = f"""
        1. Filter by budget range: {self.preferences.budget_min} to {self.preferences.budget_max}
        2. Use most recent year (2023)
        """
        
        # Build demographic conditions based on user preferences
        demographic_conditions = ""
        if self.preferences.demographics:
            demographic_conditions = "3. Consider demographic preferences:\n"
            for demo_key, demo_value in self.preferences.demographics.items():
                if demo_value:
                    if "asian" in demo_key.lower():
                        demographic_conditions += "   - Include areas with higher Asian population percentage\n"
                    if "black" in demo_key.lower():
                        demographic_conditions += "   - Include areas with higher Black population percentage\n"
                    if "white" in demo_key.lower():
                        demographic_conditions += "   - Include areas with higher White population percentage\n"
                    # Add other demographics as needed
        
        # Additional conditions based on relaxation level
        if relaxation_level == 0:
            # All constraints
            conditions = f"""
            {base_conditions}
            {demographic_conditions}
            4. Consider areas matching the property type: {self.preferences.property_type}
            5. Consider investment goal: {self.preferences.investment_goal}
            """
        elif relaxation_level == 1:
            # Relaxed constraints - keep demographics but remove property type and investment specifics
            conditions = f"""
            {base_conditions}
            {demographic_conditions}
            Note: Ignore property type, investment goal, and risk appetite constraints
            """
        else:
            # Minimal constraints - only budget and basic demographics
            conditions = f"""
            {base_conditions}
            Note: Focus only on budget range and basic demographics
            """

        prompt = f"""
        Generate a SQL query to find matching neighborhoods in Boston based on these preferences:
        {self.preferences.model_dump_json(indent=2)}
        
        Use the SUFFOLK_REAL_ESTATE_DEMOGRAPHICS table with columns:
        - ZIP_CODE
        - AVERAGE_HOUSE_VALUE
        - INCOME_PER_HOUSEHOLD
        - TOTAL_POPULATION
        - ASIAN_POP
        - BLACK_POP
        - WHITE_POP
        - TOTAL_HOUSING_UNITS
        - YEAR
        
        Requirements:
        {conditions}
        
        Return this type of JSON structure:
        {{
            "sql_query": "YOUR SQL QUERY HERE"
        }}
        
        IMPORTANT: 
        1. Return ONLY the JSON object, no additional text
        2. Make sure the SQL query includes ALL demographic columns that match user preferences
        3. For relaxation_level {relaxation_level}, focus on essential constraints only
        4. Include WHERE conditions for each demographic preference in the user's preferences
        """
        
        try:
            response = self._get_llm_response(prompt)
            if not response["success"]:
                return {
                    "success": False,
                    "error": response.get("error", "Failed to generate SQL query")
                }
            
            # Clean and validate response
            response_text = response["response"].strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            query_data = json.loads(response_text)
            required_fields = ["sql_query"]
            if not all(field in query_data for field in required_fields):
                raise ValueError("Missing required fields in response")
            
            return {
                "success": True,
                "response": json.dumps(query_data)
            }
            
        except Exception as e:
            print(f"Error generating SQL query (relaxation_level={relaxation_level}): {str(e)}")
            return {
                "success": False,
                "error": f"Query generation failed: {str(e)}"
            }

    def _get_llm_response(self, prompt: str) -> Dict[str, Any]:
        """Get response from LLM"""
        try:
            response = self.model.generate_content(prompt)
            return {
                "success": True,
                "response": response.text if hasattr(response, 'text') else str(response)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _execute_snowflake_query(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query on Snowflake and return results"""
        try:
            cursor = self.snowflake_conn.cursor()
            cursor.execute(sql_query)
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            
            # Convert results to list of dictionaries
            results_list = []
            for row in results:
                result_dict = dict(zip(columns, row))
                results_list.append(result_dict)
            
            return {
                "success": True,
                "data": results_list
            }
        except Exception as e:
            print(f"Snowflake query execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            cursor.close()

    def reset(self):
        """Reset agent state"""
        self.preferences = UserPreferences()
        self.preferences_complete = False
        
        # Close and reopen Snowflake connection
        try:
            self.snowflake_conn.close()
        except:
            pass
        
        self.snowflake_conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
            database=os.getenv('SNOWFLAKE_DATABASE'),
            schema='SUFFOLK_REAL_ESTATE_SCHEMA'
        )