from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import snowflake.connector
from dotenv import load_dotenv
import os
from datetime import datetime
import sys
from state import ConversationState, UserPreferences
from real_estate_graph import run_real_estate_graph, initialize_real_estate_graph
import asyncio
import traceback
import re
import time
# import markdown
# import pdfkit  # You'll need to install this
# Load environment variables
load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from backend.agents.query_agent import QueryAgent
from backend.agents.property_listings_agent import PropertyListingsAgent

# Create FastAPI app
app = FastAPI(title="Real Estate Investment Advisor API")

# Initialize the graph when the app starts
initialize_real_estate_graph()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
query_agent = QueryAgent()
property_agent = PropertyListingsAgent()

# Create a directory to store reports
REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Store ongoing research tasks
research_tasks = {}

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

# Chat API Models
class ChatRequest(BaseModel):
    message: str
    state: Dict[str, Any]

class ChatResponse(BaseModel):
    message: str
    state: Dict[str, Any]
    neighborhoods: List[Dict[str, Any]] | None = None
    research_task_id: Optional[str] = None

# Research API Models
class ResearchRequest(BaseModel):
    """Request model for the research endpoint."""
    query: str
    preferences: Optional[Dict[str, Any]] = None

class ResearchResponse(BaseModel):
    """Response model for the research endpoint."""
    task_id: str
    status: str
    message: str

class ResearchResult(BaseModel):
    """Result model for the research task."""
    task_id: str
    status: str
    report: Optional[str] = None
    error: Optional[str] = None
    completed_at: Optional[str] = None
    pdf_available: bool = False
    pdf_path: Optional[str] = None

# def convert_markdown_to_pdf(markdown_content, output_path):
#     """Convert markdown content to PDF and save it to the specified path"""
#     try:
#         # Convert markdown to HTML
#         html_content = markdown.markdown(markdown_content)
        
#         # Add CSS for better styling
#         styled_html = f"""
#         <html>
#         <head>
#             <style>
#                 body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
#                 h1 {{ color: #1f77b4; font-size: 28px; }}
#                 h2 {{ color: #2c3e50; font-size: 24px; margin-top: 30px; }}
#                 h3 {{ color: #34495e; font-size: 20px; }}
#                 img {{ max-width: 100%; height: auto; }}
#                 table {{ border-collapse: collapse; width: 100%; }}
#                 th, td {{ border: 1px solid #ddd; padding: 8px; }}
#                 th {{ background-color: #f2f2f2; }}
#                 code {{ background-color: #f9f9f9; padding: 2px 4px; }}
#                 pre {{ background-color: #f9f9f9; padding: 10px; overflow-x: auto; }}
#                 a {{ color: #0066cc; }}
#                 hr {{ border: 0; border-top: 1px solid #eee; margin: 20px 0; }}
#             </style>
#         </head>
#         <body>
#             {html_content}
#         </body>
#         </html>
#         """
        
#         # Convert HTML to PDF and save to file
#         pdfkit.from_string(styled_html, output_path)
#         return True
#     except Exception as e:
#         print(f"Error converting to PDF: {e}")
#         traceback.print_exc()
#         return False

def run_research_task(task_id: str, query: str, preferences: Dict[str, Any] = None):
    try:
        print("\n" + "="*80)
        print("🔍 RESEARCH TASK STARTING")
        print("="*80)
        
        # Run the research graph
        result = run_real_estate_graph(query, preferences)
        
        # Extract the report markdown and existing report file path
        report_markdown = result.get("output", "No report generated.")
        error = result.get("error")
        existing_report_file = result.get("report_file")
        
        # Use the existing report file if available
        if existing_report_file and os.path.exists(existing_report_file):
            report_path = existing_report_file
            print(f"Using existing report file: {report_path}")
            
            # Read the content from the existing file
            with open(report_path, "r", encoding="utf-8") as f:
                report_content = f.read()
                print(f"Successfully read from existing report file (Size: {len(report_content)} bytes)")
            
            # Generate a PDF filename based on markdown filename (but don't try to create it)
            base_filename = os.path.basename(report_path).replace(".md", "")
            
        else:
            # If no existing report file or it doesn't exist, create a new one
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{task_id}_{timestamp}"
            filename = f"{base_filename}.md"
            report_path = os.path.join(REPORTS_DIR, filename)
            
            # Ensure the report directory exists
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            # Clean up the report markdown before saving
            report_content = _sanitize_report(report_markdown)
            
            # Write the file with immediate flush
            with open(report_path, "w", encoding="utf-8", buffering=1) as f:
                f.write(report_content)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
            
            # Verify the file was written
            if os.path.exists(report_path):
                file_size = os.path.getsize(report_path)
                print(f"Created new report file: {report_path} (Size: {file_size} bytes)")
            else:
                print(f"Error: Report file was not created at {report_path}")
        
        # Skip PDF generation since the function is commented out
        pdf_success = False
        pdf_path = None
        
        # Update task status
        research_tasks[task_id] = {
            "task_id": task_id,
            "status": "completed" if not error else "failed",
            "report": report_content,
            "error": error,
            "report_path": report_path,
            "pdf_path": pdf_path,
            "completed_at": datetime.now().isoformat()
        }
        
        print(f"Task status updated. Report content size: {len(report_content) if report_content else 0} bytes")
        
    except Exception as e:
        print(f"Error in run_research_task: {e}")
        traceback.print_exc()
        research_tasks[task_id] = {
            "task_id": task_id,
            "status": "failed",
            "report": None,
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        }

def _sanitize_report(report_text):
    """Sanitize report text by removing unhelpful content"""
    # Patterns to remove
    patterns_to_remove = [
        r"Source from .* excluded - using only verified real estate data sources for your protection\.",
        r"Untrusted domain skipped",
        r"No content available"
    ]
    
    # Remove each pattern
    clean_text = report_text
    for pattern in patterns_to_remove:
        clean_text = re.sub(pattern, "", clean_text)
    
    # Remove empty sections or placeholders
    clean_text = re.sub(r"##### [^\n]+\n\n\n\n---\n\n", "", clean_text)
    
    # Remove consecutive newlines (more than 2)
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)
    
    return clean_text

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Handle chat messages and manage conversation state"""
    try:
        state = ConversationState(**request.state)
        state.add_message("user", request.message)
        
        # Create QueryAgent with existing preferences if any
        query_agent = QueryAgent()
        if state.preferences:
            query_agent.preferences = UserPreferences(**state.preferences)
            query_agent.preferences_complete = state.preferences_complete
        
        # Process with QueryAgent
        print("🔄 Processing user input")
        query_result = query_agent.process_input(request.message)
        print("✅ Completed processing input")
        
        # Update state with new preferences
        state.preferences = query_agent.preferences.model_dump()
        state.preferences_complete = query_agent.preferences_complete
        state.is_complete = query_agent.preferences_complete
        
        if state.is_complete:
            task_id = f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            research_tasks[task_id] = {
                "task_id": task_id,
                "status": "running",
                "report": None,
                "error": None,
                "started_at": datetime.now().isoformat()
            }
            
            print(f"Starting research task with ID: {task_id}")
            
            # Pass full conversation context and preferences
            background_tasks.add_task(
                run_research_task,
                task_id=task_id,
                query=state.get_messages_text(),
                preferences=state.preferences
            )
            
            response_message = (
                "I'm preparing a comprehensive real estate investment analysis based on your preferences. "
                "This may take a few minutes. I'll notify you when it's ready. "
                f"You can check the status using the task ID: {task_id}"
            )
            
            state.add_message("assistant", response_message)
            return ChatResponse(
                message=response_message,
                state=state.model_dump(),
                neighborhoods=None,
                research_task_id=task_id
            )
        
        # If preferences aren't complete and no zip codes, get next question
        if not query_agent.preferences_complete and not query_result.zip_codes:
            next_question = query_result.next_question or "Could you provide more details about your investment preferences?"
            state.add_message("assistant", next_question)
            return ChatResponse(
                message=next_question,
                state=state.model_dump(),
                neighborhoods=None
            )
        
        error_msg = "I couldn't find any matching neighborhoods based on your preferences."
        state.add_message("assistant", error_msg)
        return ChatResponse(
            message=error_msg,
            state=state.model_dump(),
            neighborhoods=None
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        traceback.print_exc()  # Add this to see the full error trace
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/api/research", response_model=ResearchResponse)
async def start_research(background_tasks: BackgroundTasks, request: ResearchRequest):
    """Start a research task in the background."""
    # Generate a unique task ID
    task_id = f"research_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Initialize task status
    research_tasks[task_id] = {
        "task_id": task_id,
        "status": "running",
        "report": None,
        "error": None,
        "started_at": datetime.now().isoformat()
    }
    
    # Start the research task in the background
    background_tasks.add_task(
        run_research_task,
        task_id=task_id,
        query=request.query,
        preferences=request.preferences
    )
    
    return ResearchResponse(
        task_id=task_id,
        status="running",
        message="Research task started"
    )

@app.get("/api/research/{task_id}", response_model=ResearchResult)
async def get_research_status(task_id: str, refresh: bool = False, final: bool = False):
    """Get the status of a research task."""
    if task_id not in research_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = research_tasks[task_id]
    # If this is a refresh request or final attempt, force a fresh read from the report file
    force_refresh = refresh or final
    
    # If we have no report but status is completed, or this is a forced refresh, try to find the report file
    if (task_info["status"] == "completed" and not task_info.get("report")) or force_refresh:
        try:
            # Check if we have a report path and it exists
            if "report_path" in task_info and os.path.exists(task_info["report_path"]):
                with open(task_info["report_path"], "r", encoding="utf-8") as f:
                    report_content = f.read()
                    task_info["report"] = report_content
                    print(f"Read report from {task_info['report_path']}: {len(report_content)} bytes")
            else:
                # Try to find any report file with this task ID in the reports directory
                report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
                possible_files = [f for f in os.listdir(report_dir) if task_id in f and f.endswith('.md')]
                
                if possible_files:
                    # Use the most recent file (sort by modification time)
                    report_file = os.path.join(report_dir, sorted(possible_files, 
                        key=lambda x: os.path.getmtime(os.path.join(report_dir, x)), reverse=True)[0])
                    
                    with open(report_file, "r", encoding="utf-8") as f:
                        report_content = f.read()
                        task_info["report"] = report_content
                        task_info["report_path"] = report_file
                        print(f"Found report in {report_file}: {len(report_content)} bytes")
        except Exception as e:
            print(f"Error finding report file: {e}")
    
    # Ensure report is not None for completed tasks
    if task_info["status"] == "completed" and task_info.get("report") is None:
        task_info["report"] = "Report generation completed, but content could not be retrieved."
    
    # For final requests, do a more aggressive attempt to find the report
    if final and not task_info.get("report"):
        try:
            # Check entire reports directory for any files related to this task
            report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
            for filename in os.listdir(report_dir):
                if filename.endswith(".md"):
                    try:
                        report_path = os.path.join(report_dir, filename)
                        with open(report_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # If the file was just created (within the last 5 minutes)
                            if (time.time() - os.path.getmtime(report_path) < 300):
                                print(f"Found recent report file: {report_path}")
                                task_info["report"] = content
                                task_info["report_path"] = report_path
                                break
                    except Exception as file_error:
                        print(f"Error reading file {filename}: {file_error}")
        except Exception as final_error:
            print(f"Error in final report search: {final_error}")
    
    # Update status to completed if we have a report but status is still running
    if task_info["status"] == "running" and task_info.get("report"):
        task_info["status"] = "completed"
        task_info["completed_at"] = datetime.now().isoformat()
    
    return ResearchResult(
        task_id=task_id,
        status=task_info["status"],
        report=task_info.get("report"),
        error=task_info.get("error"),
        completed_at=task_info.get("completed_at"),
        pdf_available=bool(task_info.get("pdf_path")),
        pdf_path=task_info.get("pdf_path")
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/debug/research")
async def debug_research():
    """Debug endpoint to check status of all research tasks"""
    result = {}
    
    # Add a summary of all tasks
    result["tasks_count"] = len(research_tasks)
    result["completed_count"] = sum(1 for _, task in research_tasks.items() if task.get("status") == "completed")
    result["running_count"] = sum(1 for _, task in research_tasks.items() if task.get("status") == "running")
    result["failed_count"] = sum(1 for _, task in research_tasks.items() if task.get("status") == "failed")
    
    # Get basic info for the most recent tasks (up to 5)
    recent_tasks = []
    for task_id, task_info in list(research_tasks.items())[-5:]:
        recent_tasks.append({
            "task_id": task_id,
            "status": task_info.get("status"),
            "has_report": bool(task_info.get("report")),
            "report_length": len(task_info.get("report", "")) if task_info.get("report") else 0,
            "has_report_path": bool(task_info.get("report_path")),
            "report_exists": bool(task_info.get("report_path") and os.path.exists(task_info.get("report_path"))),
            "created_at": task_info.get("started_at"),
            "completed_at": task_info.get("completed_at")
        })
    
    result["recent_tasks"] = recent_tasks
    
    # Check report directory
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    try:
        report_files = os.listdir(report_dir)
        result["report_files_count"] = len(report_files)
        result["recent_report_files"] = sorted(
            [(f, os.path.getmtime(os.path.join(report_dir, f))) for f in report_files if f.endswith(".md")],
            key=lambda x: x[1], 
            reverse=True
        )[:5]
    except Exception as e:
        result["report_dir_error"] = str(e)
    
    return result