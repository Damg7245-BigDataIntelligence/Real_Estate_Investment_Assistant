from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import snowflake.connector
from dotenv import load_dotenv
import os
import json
from state import ConversationState, UserPreferences
from llm_service import LLMService
from agents.query_agent import QueryAgent

# Load environment variables
load_dotenv()

app = FastAPI(title="Real Estate Investment Advisor API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
llm_service = LLMService()
query_agent = QueryAgent()

def get_snowflake_connection():
    """Create and return a Snowflake connection"""
    return snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema=os.getenv('SNOWFLAKE_SCHEMA')
    )

class ChatRequest(BaseModel):
    message: str
    state: Dict[str, Any]

class ChatResponse(BaseModel):
    message: str
    state: Dict[str, Any]
    neighborhoods: List[Dict[str, Any]] | None = None

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat messages and manage conversation state"""
    try:
        # Convert state dict to ConversationState object
        try:
            state = ConversationState(**request.state)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid state format: {str(e)}"
            )
        
        # Add user message to state
        state.add_message("user", request.message)
        
        # Get next question or process preferences
        llm_response = await llm_service.get_next_question(
            request.message,
            state.preferences.model_dump()
        )
        
        if not llm_response["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Error getting LLM response: {llm_response.get('error', 'Unknown error')}"
            )
        
        # Parse LLM response
        try:
            response_data = json.loads(llm_response["response"])
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid LLM response format: {str(e)}"
            )
        
        # Validate response data structure
        required_fields = ["next_question", "preferences", "is_complete"]
        missing_fields = [field for field in required_fields if field not in response_data]
        if missing_fields:
            raise HTTPException(
                status_code=500,
                detail=f"Missing required fields in LLM response: {', '.join(missing_fields)}"
            )
        
        # Update state with new preferences
        state.update_preferences(response_data["preferences"])
        state.is_complete = response_data["is_complete"]
        
        # If preferences are complete, generate SQL query and get data
        neighborhoods = None
        if state.is_complete:
            print("Generating SQL query...")
            query_result = await query_agent.generate_query(state.preferences)
            
            if not query_result["success"]:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error generating SQL query: {query_result.get('error', 'Unknown error')}"
                )
            print("Connecting to Snowflake...")
            # Execute query in Snowflake
            conn = get_snowflake_connection()
            print("Connected to Snowflake")
            cursor = conn.cursor()
            
            try:
                query_data = json.loads(query_result["response"])
                print("Query data:", query_data)
                if "sql_query" not in query_data:
                    raise HTTPException(
                        status_code=500,
                        detail="SQL query not found in query result"
                    )
                
                cursor.execute(query_data["sql_query"])
                results = cursor.fetchall()
                print("Results:", results)
                # Convert results to list of dictionaries
                columns = [desc[0] for desc in cursor.description]
                neighborhoods = [dict(zip(columns, row)) for row in results]
                print("Neighborhoods:", neighborhoods)
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error executing Snowflake query: {str(e)}"
                )
            finally:
                cursor.close()
                conn.close()
        
        if state.is_complete and neighborhoods:
            # Generate a summary of the found neighborhoods
            summary = f"Based on your preferences, I've found {len(neighborhoods)} matching neighborhoods:\n\n"
            for hood in neighborhoods:
                summary += f"ZIP Code {hood['ZIP_CODE']}:\n"
                summary += f"- Average House Value: ${hood['AVERAGE_HOUSE_VALUE']:,.2f}\n"
                summary += f"- Income per Household: ${hood['INCOME_PER_HOUSEHOLD']:,.2f}\n"
                if 'ASIAN_POP' in hood and 'TOTAL_POPULATION' in hood:
                    asian_percentage = (hood['ASIAN_POP'] / hood['TOTAL_POPULATION']) * 100
                    summary += f"- Asian Population: {hood['ASIAN_POP']:,.0f} ({asian_percentage:.1f}% of total)\n"
                summary += "\n"
            summary += "Would you like more details about any of these neighborhoods?"
            
            # Update the response message
            response_data["next_question"] = summary
        
        # Add assistant response to state
        state.add_message("assistant", response_data["next_question"])
        print("State:", state.model_dump())
        return ChatResponse(
            message=response_data["next_question"],
            state=state.model_dump(),
            neighborhoods=neighborhoods
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 