from pathlib import Path
import json
import sqlite3
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "app.db"

def get_connection():
    # Cria a pasta do banco quando necessário
    DB_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return sqlite3.connect(DB_PATH)

def init_db():
    # Cria a tabela de auditoria das análises
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER,
                created_at TEXT NOT NULL,
                rpm REAL,
                candidate_family TEXT,
                historical_support REAL,
                status TEXT NOT NULL,
                similar_events_json TEXT,
                recommendation_json TEXT,
                sources_json TEXT
            )
            """
        )

        connection.commit()

def save_analysis(result):
    # Persiste somente dados necessários para rastreabilidade
    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    similar_events = result.get(
        "similar_events",
        [],
    )

    recommendation = result.get(
        "recommendation",
    )

    sources = result.get(
        "sources",
        [],
    )

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                event_id,
                created_at,
                rpm,
                candidate_family,
                historical_support,
                status,
                similar_events_json,
                recommendation_json,
                sources_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.get("event_id"),
                created_at,
                result.get("rpm"),
                result.get("candidate_family"),
                result.get("historical_support"),
                result.get("status"),
                json.dumps(
                    similar_events,
                    ensure_ascii=False,
                ),
                json.dumps(
                    recommendation,
                    ensure_ascii=False,
                ),
                json.dumps(
                    sources,
                    ensure_ascii=False,
                ),
            ),
        )

        connection.commit()

        return cursor.lastrowid

def list_analyses(limit=10):
    # Retorna análises mais recentes
    with get_connection() as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                id,
                event_id,
                created_at,
                rpm,
                candidate_family,
                historical_support,
                status
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [
        dict(row)
        for row in rows
    ]

def main():
    print("=" * 80)
    print("FIESC - BANCO SQLITE")
    print("=" * 80)

    init_db()

    print("\nBanco inicializado:")
    print(DB_PATH)

    print("\nAnálises registradas:")

    analyses = list_analyses()

    if not analyses:
        print("Nenhuma análise registrada.")
    else:
        for analysis in analyses:
            print(analysis)

if __name__ == "__main__":
    main()