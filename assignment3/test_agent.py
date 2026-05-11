import pytest

from sql_agent import agent
from sql_tools import setup_database
from utils import collect_tools
from judge import assert_criteria, evaluate_agent_performance


async def test_more_than_five_passengers():
    setup_database()

    result = await agent.run("How many trips had more than 5 passengers?")
    output = result.output

    assert output.sql_query
    assert "22413" in output.result_text.replace(",", "")


async def test_agent_gets_schema_before_running_sql():
    setup_database()

    result = await agent.run("What is the most common payment type?")
    tools = collect_tools(result.all_messages())

    assert tools[0] == "get_schema"
    assert "run_sql" in tools


async def test_highest_average_fare_hour_with_llm_judge():
    setup_database()
    question = "Which hour of the day has the highest average fare amount?"

    result = await agent.run(question)
    evaluation = await evaluate_agent_performance(
        question=question,
        answer=result.output.model_dump_json(),
        criteria=[
            "the SQL query correctly calculates average fare by hour of day",
            "the result identifies a specific hour as having the highest average fare",
            "the result includes the actual average fare amount",
        ],
    )

    assert_criteria(evaluation)


@pytest.mark.parametrize(
    ("question", "columns", "expected_text"),
    [
        (
            "What is the average tip amount for credit card payments?",
            ["tip_amount", "payment_type"],
            "4.169",
        ),
        (
            "Which pickup location (PULocationID) has the most trips?",
            ["PULocationID"],
            "132",
        ),
        (
            "What is the average fare for trips longer than 10 miles?",
            ["fare_amount", "trip_distance"],
            "62.880",
        ),
        (
            "How many trips had zero passengers recorded?",
            ["passenger_count"],
            "31465",
        ),
        (
            "What is the busiest day of the week for taxi trips?",
            ["tpep_pickup_datetime"],
            "495032",
        ),
    ],
)
async def test_additional_agent_questions(question, columns, expected_text):
    setup_database()

    result = await agent.run(question)
    output = result.output
    tools = collect_tools(result.all_messages())
    sql_query = output.sql_query.lower()
    result_text = output.result_text.replace(",", "")

    assert tools[0] == "get_schema"
    assert "run_sql" in tools
    for column in columns:
        assert column.lower() in sql_query
    assert expected_text in result_text
