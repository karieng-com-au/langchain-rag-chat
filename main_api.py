import os
from typing import List

import chromadb
import psycopg2
from chainlit.utils import mount_chainlit
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI()
DB_FILE = "./chroma"
NUTRITION_COLLECTION_NAME="nutrition_db"
QA_COLLECTION_NAME = "qas_db"

class FoodCalorie(BaseModel):
    foodcategory: str
    fooditem: str
    per100grams: str
    cals_per100grams: str
    kj_per100grams: str

@app.get("/app")
def read_main():
    return {"message": "Hello from FastAPI main app"}

@app.get("/chroma-calories")
def get_calories_info_chroma(food_name: str) -> List[dict]:
    chroma_client = chromadb.PersistentClient(DB_FILE)
    collection = chroma_client.get_collection(name=NUTRITION_COLLECTION_NAME)
    results = collection.query(query_texts=[food_name],
                               n_results=5)
    return results["metadatas"][0] if results["metadatas"] else []


@app.get("/qa")
def get_answer(question: str) -> List[dict]:
    chroma_client = chromadb.PersistentClient(DB_FILE)
    qa_collection = chroma_client.get_collection(name=QA_COLLECTION_NAME)
    results = qa_collection.query(query_texts=[question], n_results=5)
    return results

@app.get("/calories")
def get_calories_info(food_name: str) -> List[FoodCalorie]:
    params = {
    "host": os.getenv("POSTGRES_HOST"),
    "database": os.getenv("POSTGRES_DATABASE"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "port": os.getenv("POSTGRES_PORT")
    }
    conn = psycopg2.connect(**params)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM calories WHERE LOWER(fooditem) = %s", (str.lower(food_name),))
            rows = cur.fetchall()
        conn.commit()
        if not rows:
            return []
        return [
            FoodCalorie(
                foodcategory=foodcategory,
                fooditem=fooditem,
                per100grams=per100grams,
                cals_per100grams=cals_per100grams,
                kj_per100grams=kj_per100grams,
            )
            for foodcategory, fooditem, per100grams, cals_per100grams, kj_per100grams in rows
        ]
    finally:
        conn.close()

mount_chainlit(app=app, target="main.py", path="/")
