# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from run_agent import agent_executor

app = FastAPI()

class Query(BaseModel):
    question: str

@app.post("/api/query")
async def query_agent(q: Query):
    response = agent_executor.invoke({"input": q.question})
    return {"answer": response["output"]}