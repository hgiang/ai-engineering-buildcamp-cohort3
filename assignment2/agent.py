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
TIMEOUT = 30

MODEL = "openai:gpt-4o-mini"

_session = requests.Session()
_session.headers.update(HEADERS)


def search_wikipedia(query: str) -> dict:
    params = {"action": "query", "format": "json", "list": "search", "srsearch": query}
    response = _session.get(SEARCH_API, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def get_page(title: str) -> str:
    params = {"title": title, "action": "raw"}
    response = _session.get(PAGE_API, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text


def _get_search_results(query: str) -> list[dict]:
    return search_wikipedia(query)["query"]["search"]


_agent = Agent(system_prompt="You are a helpful assistant. Use the provided tools to answer questions.")


@_agent.tool_plain
def search_wikipedia_tool(query: str) -> dict:
    return search_wikipedia(query)


@_agent.tool_plain
def get_page_tool(title: str) -> str:
    return get_page(title)


class WikipediaAgent:
    def _run_agent(self, question: str) -> tuple[str, list[str]]:
        result = _agent.run_sync(question, model=MODEL)
        messages = result.all_messages()
        tool_calls = [
            f"{part.tool_name}({part.args})"
            for msg in messages
            for part in getattr(msg, "parts", [])
            if getattr(part, "part_kind", None) == "tool-call"
        ]
        return result.output, tool_calls

    def count_search_results(self, query: str = "capybara") -> None:
        print(f"Total results: {len(_get_search_results(query))}")

    def count_matching_titles(self, query: str = "capybara") -> None:
        query_lower = query.lower()
        count = 0
        for r in _get_search_results(query):
            title = r["title"]
            matches = query_lower in title.lower()
            if matches:
                count += 1
            print(f"  {'✓' if matches else '✗'} {title}")
        print(f"Titles containing '{query}': {count}")

    def measure_page_length(self, title: str = "Capybara") -> None:
        print(f"Character count: {len(get_page(title))}")

    def agent_info(self) -> None:
        print(f"Framework: PydanticAI")
        print(f"LLM provider: {MODEL}")

    def summarize_page(self, url: str = "https://en.wikipedia.org/wiki/Capybara") -> None:
        answer, _ = self._run_agent(f"What is this page about? {url}")
        print(answer)

    def research_threats(self, question: str = "What are the main threats to capybara populations?") -> None:
        answer, tool_calls = self._run_agent(question)
        print(f"Tool calls ({len(tool_calls)} total): {' -> '.join(tool_calls)}")
        print()
        print(answer)


def main() -> None:
    fire.Fire(WikipediaAgent)


if __name__ == "__main__":
    main()
