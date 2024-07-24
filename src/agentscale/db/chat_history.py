import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List

import asyncpg

logger = logging.getLogger(__name__)

DB_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@pgvector:5432/postgres"
)


async def initialize_database():
    try:
        conn = await asyncpg.connect(DB_URL)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                conversation_id TEXT,
                message TEXT,
                role TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        await conn.close()
        logger.info("Initialized chat history table in PostgreSQL database")
    except asyncpg.PostgresError as e:
        logger.error(f"Database error during initialization: {e}")


async def save_chat_history(conversation_id: str, message: str, role: str):
    try:
        conn = await asyncpg.connect(DB_URL)
        await conn.execute(
            "INSERT INTO chat_history (conversation_id, message, role, timestamp) VALUES ($1, $2, $3, $4)",
            conversation_id,
            message,
            role,
            datetime.now(),
        )
        await conn.close()
        logger.debug(
            f"Saved message to chat history for conversation {conversation_id}"
        )
    except asyncpg.PostgresError as e:
        logger.error(f"Database error while saving chat history: {e}")


async def get_chat_history(
    conversation_id: str, limit: int = 10
) -> List[Dict[str, str]]:
    try:
        conn = await asyncpg.connect(DB_URL)
        rows = await conn.fetch(
            """
            SELECT message, role, timestamp 
            FROM chat_history 
            WHERE conversation_id = $1 
            ORDER BY timestamp DESC
            LIMIT $2
            """,
            conversation_id,
            limit,
        )
        await conn.close()
        history = [dict(row) for row in rows]
        logger.debug(
            f"Retrieved {len(history)} messages for conversation {conversation_id}"
        )
        return list(reversed(history))  # Reverse to maintain chronological order
    except asyncpg.PostgresError as e:
        logger.error(f"Database error while retrieving chat history: {e}")
        return []


# Call this function at the start of your application
async def main():
    await initialize_database()


# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
