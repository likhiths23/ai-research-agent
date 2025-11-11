# run_agent.py
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

# ---- Setup LLM ----
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",  # Updated model
    temperature=0.3
)

# ---- Define Tools ----
tools = [
    Tool(
        name="PDF_Retriever",
        func=pdf_search_tool,
        description="Retrieve relevant content from research PDFs. Use this when you need information from uploaded documents."
    ),
    Tool(
        name="Web_Search",
        func=web_search_tool,
        description="Search the web (Google Scholar) for recent papers. Use this when you need current research or papers not in the PDF database."
    ),
    Tool(
        name="Citation_Generator",
        func=lambda text: citation_tool(title=text),
        description="Generate formatted citations for research content. Input should be the title of the paper."
    )
]

# ---- Create Prompt Template ----
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

# ---- Initialize Agent ----
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=10
)

if __name__ == "__main__":
    print("🤖 AI Research Agent (Groq-powered)")
    print("-----------------------------------")
    print("Available models: llama-3.1-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it")
    print()
    while True:
        query = input("\nAsk your research question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            response = agent_executor.invoke({"input": query})
            print("\n🔍 Response:\n", response["output"])
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try rephrasing your question.")