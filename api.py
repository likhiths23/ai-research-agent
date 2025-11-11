# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from tools.pdf_tool import pdf_search_tool
from tools.web_tool import web_search_tool
from tools.citation_tool import citation_tool

load_dotenv()

app = FastAPI(title="AI Research Agent API")

# CORS middleware - allows frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Lovable domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM and Agent
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.3
)

tools = [
    Tool(
        name="PDF_Retriever",
        func=pdf_search_tool,
        description="Retrieve relevant content from research PDFs."
    ),
    Tool(
        name="Web_Search",
        func=web_search_tool,
        description="Search Google Scholar for recent papers."
    ),
    Tool(
        name="Citation_Generator",
        func=lambda text: citation_tool(title=text),
        description="Generate formatted citations."
    )
]

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

# Request/Response Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    success: bool

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "AI Research Agent API",
        "status": "running",
        "endpoints": {
            "/query": "POST - Submit research questions",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API is running"}

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    try:
        if not request.question or len(request.question.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        response = agent_executor.invoke({"input": request.question})
        
        return QueryResponse(
            answer=response["output"],
            success=True
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)