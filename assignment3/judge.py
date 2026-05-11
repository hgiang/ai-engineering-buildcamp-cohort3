from pydantic import BaseModel
from pydantic_ai import Agent


class Evaluation(BaseModel):
    passes: bool
    feedback: str


judge_agent = Agent(
    "openai:gpt-4o-mini",
    output_type=Evaluation,
    instructions=(
        "You are grading a SQL agent response. "
        "Check only the provided agent answer against every criterion. "
        "Do not require the judge feedback itself to contain anything. "
        "Tabular values count as identifying a result when the SQL/result columns make their meaning clear. "
        "Return passes=true only when all criteria are satisfied."
    ),
)


async def evaluate_agent_performance(question, answer, criteria):
    criteria_text = "\n".join(f"- {criterion}" for criterion in criteria)
    prompt = f"""
Question:
{question}

Agent answer:
{answer}

Criteria:
{criteria_text}
""".strip()

    result = await judge_agent.run(prompt)
    return result.output


def assert_criteria(evaluation):
    assert evaluation.passes, evaluation.feedback
