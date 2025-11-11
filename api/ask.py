import json
import os
from run_agent import agent_executor

def handler(request):
    try:
        body = json.loads(request.body)
        query = body.get("question", "")
        if not query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'question' field"})
            }
        response = agent_executor.invoke({"input": query})
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"answer": response["output"]})
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
