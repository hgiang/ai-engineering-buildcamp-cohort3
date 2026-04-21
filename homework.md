# AI Engineering Buildcamp - Week 1 Homework

In this homework we'll practice working with documents, extracting text, and preparing data for AI applications.

We will work with books from Allen Downey (Green Tea Press). These books are free to download, which makes them perfect for this homework.

We will extract text from the PDF files, chunk them, and use them to build a RAG system.

When submitting your homework, you will also need to include a link to your GitHub repository or other public code-hosting site.

Note: For all questions, if your answer doesn't match exactly due to differences in tools or environments, pick the closest option.

## Downloading the Books

Download the CSV file `books.csv` with book links:

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/ai-engineering-buildcamp-code/main/01-foundation/homework/books.csv
```

Write a script to download all the PDFs from the CSV file. You can save them anywhere you want. For example, the `books/` directory.

Ask AI to help if you don't want to write this code yourself.

## Question 1. Converting PDFs to Markdown

We want now to extract text from these books.

Install the `markitdown` library with PDF support:

```bash
uv add 'markitdown[pdf]'
```

This library can convert various document formats including PDF to markdown or text format.

Convert all the downloaded PDFs to markdown files and save them to a `books_text/` directory.

How many lines are in the extracted content from the "Think Python" book?

- 12,268
- 14,268
- 16,268
- 18,268

Hint: you can use `wc -l` or just open it with VS Code.

## Question 2. Chunking for RAG

For RAG we need to split documents into smaller chunks.

First, prepare your documents:

- Read each markdown file from your `books_text/` directory
- Split the content into lines
- Remove empty lines and lines that contain only whitespace
- Turn each book into a dictionary with `source` (filename) and `content` (list of non-empty lines)

After that, chunk it.

Use the `gitsource` package which provides the `chunk_documents` function. Install it if you don't have it:

```bash
uv add gitsource
```

The `chunk_documents` function uses a sliding window approach with these parameters:

- `size=100`: number of items per chunk
- `step=50`: how many items to move forward for each chunk

With `size=100` and `step=50`, each chunk is about 4,400 characters or 780 words on average.

How many chunks are produced for the "Think Python" book with these settings?

- 134
- 214
- 294
- 374

## Question 3. Indexing with minsearch

Now we need to index our chunks so we can search through them.

We'll use `minsearch`. Install it if you don't have it:

```bash
uv add minsearch
```

Load all your chunked documents and create an index:

```python
from minsearch import Index

documents = prepare_documents(chunks)
# here you need to turn the lists into strings
# e.g. with content = "\n".join(chunk["content"])

index.fit(documents)
```

How many documents (chunks) did you index?

- 719
- 919
- 1119
- 1319

## Question 4. Searching and RAG

Now let's search our index.

```python
results = index.search("python function definition", num_results=5)
```

Look at the top result. Which book did it come from?

- Think Python
- Think DSP
- Think Java
- Think Complexity

## Question 5. Full RAG

We're ready to do RAG.

This is the code we wrote in the lessons (or similar to it - but please use the code below to make sure the results are reproducible):

```python
import json

instructions = """
You're a course assistant, your task is to answer the QUESTION from the
course students using the provided CONTEXT
"""

prompt_template = """
<QUESTION>
{question}
</QUESTION>

<CONTEXT>
{context}
</CONTEXT>
""".strip()

def build_prompt(question, search_results):
    context = json.dumps(search_results, indent=2)
    prompt = prompt_template.format(
        question=question,
        context=context
    ).strip()
    return prompt

def search(question):
    return index.search(question, num_results=5)

def llm(user_prompt, instructions, model='gpt-4o-mini'):
    messages = [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = openai_client.responses.create(
        model=model,
        input=messages
    )

    return response.output_text

def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, instructions)
    return answer
```

Do RAG for `"python function definition"`. What's the response?

You don't have to use OpenAI, you can use an alternative provider.

Now let's change these functions to also return the number of input and output tokens.

How many input tokens did we use for this one RAG query?

- 4889
- 6889
- 8889
- 10889

## Question 6. Structured vs Unstructured Output

Now let's use structured outputs with a Pydantic model.

Define the response model:

```python
from pydantic import BaseModel, Field
from typing import Literal

class RAGResponse(BaseModel):
    answer: str = Field(description="The main answer to the user's question in markdown")
    found_answer: bool = Field(description="True if relevant information was found in the documentation")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    confidence_explanation: str = Field(description="Explanation about the confidence level")
    answer_type: Literal["how-to", "explanation", "troubleshooting", "comparison", "reference"] = Field(description="The category of the answer")
    followup_questions: list[str] = Field(description="Suggested follow-up questions")
```

Modify the `llm` and `rag` functions from Question 5 to use structured outputs.

- do RAG for `"python function definition"`
- look at the number of input tokens
- compare the number with the results from Q5

How many MORE input tokens does the structured output version use compared to the unstructured version?

- 24
- 224
- 424
- 624

## AI Assistants

You can use AI to solve this homework. But make sure you understand every step.

## Submitting the Solutions

Form for submitting: https://courses.datatalks.club/ai-buildcamp-3/homework/hw12

## Learning in Public

We encourage everyone to share what they learned. This is called "learning in public".

Why learn in public?

- Accountability: Sharing your progress creates commitment and motivation to continue
- Feedback: The community can provide valuable suggestions and corrections
- Networking: You'll connect with like-minded people and potential collaborators
- Documentation: Your posts become a learning journal you can reference later
- Opportunities: Employers and clients often discover talent through public learning

You can read more about the benefits here.

Don't worry about being perfect. Everyone starts somewhere, and people love following genuine learning journeys.

### Example post for LinkedIn

🚀 Week 1 of AI Engineering Buildcamp complete!

Just finished the RAG (Retrieval Augmented Generation) module. Learned how to:

✅ Process PDF documents - all Allen B. Downey's books  
✅ Extract markdown with markitdown  
✅ Chunk documents for better retrieval  
✅ Build a full RAG pipeline

Here's my homework solution: `<LINK>`

Thanks to Alexey Grigorev for this amazing content - who else is learning RAG?

Check out the course: https://maven.com/alexey-grigorev/from-rag-to-agents

### Example post for Twitter/X

🤖 RAG module complete!

Just finished the RAG module:
- Processed all Allen B. Downey's books
- Extracted markdown with markitdown
- Chunked documents for retrieval
- Built a full RAG pipeline

My solution: `<LINK>`

Thanks @Al_Grigor!

Course: https://maven.com/alexey-grigorev/from-rag-to-agents
