# PostgreSQL stores user accounts, OAuth session data, and workspace settings
"""Database connection helper."""

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL) if DATABASE_URL else None


def test_connection() -> None:
    """Test the database connection with a simple query."""
    if not engine:
        print("Error: DATABASE_URL is not set.")
        return

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        print("Database connection successful!")
    except Exception as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    test_connection()
