import os
from serpapi import GoogleSearch

def web_search_tool(query: str, num=3):
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "num": num
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    output = []
    for r in results:
        title = r.get("title")
        link = r.get("link")
        snippet = r.get("snippet")
        output.append(f"{title}\n{snippet}\n{link}")
    return "\n\n".join(output)
    