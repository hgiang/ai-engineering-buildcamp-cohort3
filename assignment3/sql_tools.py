import os
import urllib.request

import duckdb

DATA_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet"
PARQUET_FILE = "yellow_tripdata_2024-01.parquet"
con = duckdb.connect("taxi.db")


def setup_database() -> int:
    """Download the parquet file and load it into DuckDB."""
    if not os.path.exists(PARQUET_FILE):
        print(f"Downloading {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, PARQUET_FILE)

    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS trips AS
        SELECT * FROM '{PARQUET_FILE}'
        """
    )
    count = con.execute("SELECT COUNT(*) FROM trips").fetchone()[0]
    print(f"Loaded {count} rows")
    return count


class SQLTools:
    def get_schema(self) -> str:
        rows = con.execute("DESCRIBE trips").fetchall()
        return "\n".join(f"{name}: {data_type}" for name, data_type, *_ in rows)

    def run_sql(self, query: str) -> str:
        result = con.execute(query)
        columns = [column[0] for column in result.description]
        rows = result.fetchmany(50)

        lines = [" | ".join(columns)]
        for row in rows:
            lines.append(" | ".join(str(value) for value in row))

        return "\n".join(lines)


if __name__ == "__main__":
    setup_database()
