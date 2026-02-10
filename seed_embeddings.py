import os

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

params = {
    "host": os.getenv("POSTGRES_HOST"),
    "database": os.getenv("POSTGRES_DATABASE"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": os.getenv("POSTGRES_PORT"),
}

conn = psycopg2.connect(**params)
try:
    with conn.cursor() as cur:
        cur.execute("SELECT ctid, foodcategory, fooditem FROM calories")
        rows = cur.fetchall()

        if not rows:
            print("No rows found in calories table.")
        else:
            # Build texts, replacing None with empty string
            texts = [f"{cat or ''} {item or ''}".strip() for _, cat, item in rows]

            print(f"Found {len(rows)} rows. Sample: {texts[:3]}")

            # Embed in batches of 100 (API limit is 2048)
            batch_size = 100
            for i in range(0, len(rows), batch_size):
                batch_rows = rows[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]

                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch_texts,
                )

                for row, embedding_obj in zip(batch_rows, response.data):
                    ctid = row[0]
                    vector = embedding_obj.embedding
                    cur.execute(
                        "UPDATE calories SET embedding = %s::vector WHERE ctid = %s",
                        (str(vector), ctid),
                    )

                print(f"  Embedded batch {i // batch_size + 1} ({len(batch_texts)} rows)")

    conn.commit()
    print(f"Done. Embedded {len(rows)} rows total.")
finally:
    conn.close()
