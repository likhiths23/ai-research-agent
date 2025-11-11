# api.py
import os
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_groq import ChatGroq

from tools.pdf_tool import pdf_search_tool
from tools.web_tool import web_search_tool
from tools.citation_tool import citation_tool

load_dotenv()

app = FastAPI()

# Allow frontend (Lovable/Vercel) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize LLM & Tools ---
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.3,
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
        description="Generate citations for papers."
    )
]

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the format:

Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class Query(BaseModel):
    question: str

@app.post("/query")
def query_agent(data: Query):
    try:
        result = agent_executor.invoke({"input": data.question})
        return {"response": result["output"]}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-pdfs")
async def upload_pdfs(files: list[UploadFile]):
    import tempfile, shutil, subprocess
    pdf_paths = []
    for file in files:
        tmp_path = os.path.join(tempfile.gettempdir(), file.filename)
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        pdf_paths.append(tmp_path)

    subprocess.run(["python", "ingest_pdfs.py", *pdf_paths])
    return {"status": "PDFs ingested successfully!"}
