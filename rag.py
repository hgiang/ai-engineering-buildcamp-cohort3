"""
Week 1 RAG pipeline CLI. Subcommands map to homework steps.

Examples:
  uv run python rag.py download-books --books-dir books
  uv run python rag.py pdf-to-markdown --books-dir books --out-dir books_text
  uv run python rag.py chunk --books-text-dir books_text
  uv run python rag.py index --books-text-dir books_text
  uv run python rag.py search-rag --query "..."
  uv run python rag.py full-rag --query "..."
  uv run python rag.py compare-structured-unstructured --query "..."
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlparse

import fire
import requests
from gitsource import chunk_documents
from markitdown import MarkItDown
from minsearch import Index
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

ROOT = Path(__file__).resolve().parent


class RAGResponse(BaseModel):
    answer: str = Field(description="The main answer to the user's question in markdown")
    found_answer: bool = Field(description="True if relevant information was found in the documentation")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    confidence_explanation: str = Field(description="Explanation about the confidence level")
    answer_type: Literal["how-to", "explanation", "troubleshooting", "comparison", "reference"] = Field(description="The category of the answer")
    followup_questions: list[str] = Field(description="Suggested follow-up questions")

BOOKS_CSV_URL = (
    "https://raw.githubusercontent.com/alexeygrigorev/ai-engineering-buildcamp-code/"
    "main/01-foundation/homework/books.csv"
)

INSTRUCTIONS = """
You're a course assistant, your task is to answer the QUESTION from the
course students using the provided CONTEXT
"""

PROMPT_TEMPLATE = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()


class RAGPipeline:
    """Entry points for each pipeline stage."""

    def download_books(
        self,
        books_dir: str = "books",
        csv_url: str = BOOKS_CSV_URL,
    ) -> None:
        """1. Fetch books.csv from csv_url, then download each pdf_url into books_dir."""
        dest = ROOT / books_dir
        dest.mkdir(parents=True, exist_ok=True)
        csv_path = dest / "books.csv"

        r = requests.get(csv_url, timeout=120)
        r.raise_for_status()
        csv_path.write_bytes(r.content)

        with csv_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                url = row["pdf_url"].strip()
                name = Path(urlparse(url).path).name
                out = dest / name
                if out.exists() and out.stat().st_size > 0:
                    continue
                pdf = requests.get(url, timeout=300)
                pdf.raise_for_status()
                out.write_bytes(pdf.content)

    def pdf_to_markdown(
        self,
        books_dir: str = "books",
        out_dir: str = "books_text",
        thinkpython_only: bool = False,
    ) -> None:
        """2. Convert PDFs under books_dir to Markdown under out_dir (markitdown)."""
        src = ROOT / books_dir
        dest = ROOT / out_dir
        dest.mkdir(parents=True, exist_ok=True)

        paths = [src / "thinkpython2.pdf"] if thinkpython_only else sorted(src.glob("*.pdf"))
        for path in paths:
            result = MarkItDown().convert(str(path))
            out = dest / path.with_suffix(".md").name
            out.write_text(result.text_content, encoding="utf-8")
            print(f"Converted {path.name} to {out.name}")

        if thinkpython_only:
            text = (dest / "thinkpython2.md").read_text(encoding="utf-8")
            print("Line count (same idea as wc -l):", text.count("\n"))

    def chunk(
        self,
        books_text_dir: str = "books_text",
        size: int = 100,
        step: int = 50,
        thinkpython_only: bool = False,
    ) -> list:
        """3. Load markdown, normalize lines, chunk with gitsource.chunk_documents."""
        base = ROOT / books_text_dir
        paths = [base / "thinkpython2.md"] if thinkpython_only else sorted(base.glob("*.md"))

        documents = []
        for path in paths:
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
            documents.append({"source": path.name, "content": lines})

        chunks = chunk_documents(documents, size=size, step=step)
        print(f"Total chunks: {len(chunks)}")
        return chunks

    def index(
        self,
        books_text_dir: str = "books_text",
        size: int = 100,
        step: int = 50,
    ) -> Index:
        """4. Chunk documents and build a minsearch.Index (BM25)."""
        chunks = self.chunk(books_text_dir=books_text_dir, size=size, step=step)
        documents = [{"source": c["source"], "content": "\n".join(c["content"])} for c in chunks]

        idx = Index(text_fields=["content"], keyword_fields=["source"])
        idx.fit(documents)
        print(f"Indexed {len(documents)} documents")
        return idx

    def search_rag(
        self,
        query: str,
        books_text_dir: str = "books_text",
        size: int = 100,
        step: int = 50,
        top_k: int = 5,
    ) -> None:
        """5. Search index and print the top result's source book (Q4)."""
        idx = self.index(books_text_dir=books_text_dir, size=size, step=step)
        results = idx.search(query, num_results=top_k)
        print(f"Top result source: {results[0]['source']}")

    def full_rag(
        self,
        query: str,
        books_text_dir: str = "books_text",
        size: int = 100,
        step: int = 50,
        top_k: int = 5,
    ) -> None:
        """6. End-to-end: chunk/index/search + LLM answer using course prompt template."""
        idx = self.index(books_text_dir=books_text_dir, size=size, step=step)
        results = idx.search(query, num_results=top_k)

        prompt = PROMPT_TEMPLATE.format(question=query, context=json.dumps(results, indent=2))

        response = OpenAI().responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ],
        )
        print(response.output_text)
        print(f"\nInput tokens: {response.usage.input_tokens}, Output tokens: {response.usage.output_tokens}")

    def compare_structured_unstructured(
        self,
        query: str,
        books_text_dir: str = "books_text",
        size: int = 100,
        step: int = 50,
        top_k: int = 5,
    ) -> None:
        """7. Same RAG context; compare plain text vs responses.parse (Pydantic) output."""
        idx = self.index(books_text_dir=books_text_dir, size=size, step=step)
        results = idx.search(query, num_results=top_k)
        prompt = PROMPT_TEMPLATE.format(question=query, context=json.dumps(results, indent=2))
        messages = [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ]

        client = OpenAI()

        unstructured = client.responses.create(model="gpt-4o-mini", input=messages)
        structured = client.responses.parse(model="gpt-4o-mini", input=messages, text_format=RAGResponse)

        print(f"Unstructured input tokens: {unstructured.usage.input_tokens}")
        print(f"Structured   input tokens: {structured.usage.input_tokens}")
        print(f"Difference: {structured.usage.input_tokens - unstructured.usage.input_tokens}")


def main() -> None:
    fire.Fire(RAGPipeline)


if __name__ == "__main__":
    main()
