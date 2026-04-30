"""
Week 3 Wikipedia Agent CLI. Subcommands map to homework questions.

Examples:
  uv run python agent.py count-search-results
  uv run python agent.py count-matching-titles
  uv run python agent.py measure-page-length
  uv run python agent.py agent-info
  uv run python agent.py summarize-page
  uv run python agent.py research-threats
"""

from __future__ import annotations

import requests
import fire
from pydantic_ai import Agent

HEADERS = {"User-Agent": "ai-engineering-buildcamp/1.0 (educational project)"}
SEARCH_API = "https://en.wikipedia.org/w/api.php"
PAGE_API = "https://en.wikipedia.org/w/index.php"

MODEL = "gpt-4o-mini"


def search_wikipedia(query: str) -> dict:
    """Call Wikipedia Search API and return the raw JSON response."""
    params = {"action": "query", "format": "json", "list": "search", "srsearch": query}
    response = requests.get(SEARCH_API, params=params, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def get_page(title: str) -> str:
    """Fetch raw content of a Wikipedia page by title."""
    params = {"title": title, "action": "raw"}
    response = requests.get(PAGE_API, params=params, headers=HEADERS)
    response.raise_for_status()
    return response.text


_agent = Agent(system_prompt="You are a helpful assistant. Use the provided tools to answer questions.")


@_agent.tool_plain
def search_wikipedia_tool(query: str) -> dict:
    """Search Wikipedia for pages related to a query."""
    return search_wikipedia(query)


@_agent.tool_plain
def get_page_tool(title: str) -> str:
    """Fetch the raw content of a Wikipedia page by title."""
    return get_page(title)


class WikipediaAgent:
    def _run_agent(self, question: str) -> tuple[str, list[str]]:
        """Run the PydanticAI agent and return (answer, tool_call_log)."""
        result = _agent.run_sync(question, model=f"openai:{MODEL}")
        tool_calls = [
            f"{part.tool_name}({part.args})"
            for msg in result.all_messages()
            for part in getattr(msg, "parts", [])
            if getattr(part, "part_kind", None) == "tool-call"
        ]
        return result.output, tool_calls

    def count_search_results(self, query: str = "capybara") -> None:
        """Q1: Search Wikipedia for query and print the total number of results."""
        results = search_wikipedia(query)
        count = len(results["query"]["search"])
        print(f"Total results: {count}")

    def count_matching_titles(self, query: str = "capybara") -> None:
        """Q2: Search for query and count how many result titles contain the query term."""
        results = search_wikipedia(query)
        titles = [r["title"] for r in results["query"]["search"]]
        count = sum(1 for t in titles if query.lower() in t.lower())
        print(f"Titles containing '{query}': {count}")
        for t in titles:
            print(f"  {'✓' if query.lower() in t.lower() else '✗'} {t}")

    def measure_page_length(self, title: str = "Capybara") -> None:
        """Q3: Fetch a Wikipedia page by title and print the character count."""
        content = get_page(title)
        print(f"Character count: {len(content)}")

    def agent_info(self) -> None:
        """Q4: Print the framework and LLM provider used by this agent."""
        print("Framework: PydanticAI")
        print(f"LLM provider: OpenAI ({MODEL})")

    def summarize_page(self, url: str = "https://en.wikipedia.org/wiki/Capybara") -> None:
        """Q5: Ask the agent what a Wikipedia page is about."""
        answer, _ = self._run_agent(f"What is this page about? {url}")
        print(answer)

    def research_threats(self, question: str = "What are the main threats to capybara populations?") -> None:
        """Q6: Ask the agent a research question; print tool call log and answer."""
        answer, tool_calls = self._run_agent(question)
        print(f"Tool calls ({len(tool_calls)} total): {' -> '.join(tool_calls)}")
        print()
        print(answer)


def main() -> None:
    fire.Fire(WikipediaAgent)


if __name__ == "__main__":
    main()
