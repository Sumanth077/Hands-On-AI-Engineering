import wikipedia
from smolagents import Tool


class WikipediaTool(Tool):
    name = "wikipedia"
    description = (
        "Searches Wikipedia and returns a summary of the most relevant article "
        "for the given query. Use this for factual questions about people, places, "
        "events, concepts, or any topic likely to have a Wikipedia page."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "The topic or question to look up on Wikipedia.",
        }
    }
    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            results = wikipedia.search(query, results=3)
            if not results:
                return f"No Wikipedia results found for '{query}'."

            try:
                page = wikipedia.page(results[0], auto_suggest=False)
                summary = wikipedia.summary(results[0], sentences=5, auto_suggest=False)
                return f"Title: {page.title}\n\n{summary}\n\nURL: {page.url}"

            except wikipedia.exceptions.DisambiguationError as e:
                if e.options:
                    try:
                        summary = wikipedia.summary(e.options[0], sentences=5, auto_suggest=False)
                        return f"Title: {e.options[0]}\n\n{summary}"
                    except Exception:
                        pass
                return (
                    f"'{query}' is ambiguous. Possible topics: "
                    + ", ".join(e.options[:5])
                )

        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for '{query}'."
        except Exception as e:
            return f"Wikipedia lookup error: {e}"
