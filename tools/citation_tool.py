def citation_tool(title: str, authors=None, year=None):
    authors_str = ", ".join(authors) if authors else "Unknown"
    return f"{authors_str} ({year if year else 'n.d.'}). {title}. Retrieved from Research Agent."
