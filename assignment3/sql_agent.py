from pydantic import BaseModel
from pydantic_ai import Agent

from sql_tools import SQLTools, setup_database


class SQLResult(BaseModel):
    sql_query: str
    result_text: str
    row_count: int


sql_tools = SQLTools()

agent = Agent(
    "openai:gpt-4o-mini",
    output_type=SQLResult,
    tools=[sql_tools.get_schema, sql_tools.run_sql],
    instructions=(
        "You are a SQL agent for a DuckDB table named trips. "
        "Always call get_schema first. Then call run_sql with a valid SQL query. "
        "Return the SQL query you used, the result text, and the number of result rows. "
        "For result_text, copy the full text returned by run_sql, including column headers."
    ),
)


if __name__ == "__main__":
    setup_database()
    result = agent.run_sync("What's the average trip distance for rides with 2 passengers?")
    print(result.output)
