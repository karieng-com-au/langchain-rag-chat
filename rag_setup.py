
from typing import Dict, List

import chromadb
import pandas as pd


def prepare_nutrition_documents(csv_path: str) -> Dict:
    """
    Convert nutrition CSV into ChromaDB-ready documents.
    Each food item becomes a searchable document.
    """
    df = pd.read_csv(csv_path)

    documents = []
    metadatas = []
    ids = []

    for index, row in df.iterrows():
        # Create rich document text for semantic search
        row["Cals_per100grams"] = row["Cals_per100grams"].replace(" cal", "")
        row["KJ_per100grams"] = row["KJ_per100grams"].replace(" kJ", "")
        document_text = f"""
        Food: {row['FoodItem']}
        Category: {row['FoodCategory']}
        Nutritional Information:
        - Calories: {row['Cals_per100grams']} per 100g
        - Energy: {row['KJ_per100grams']} kJ per 100g
        - Serving size reference: {row['per100grams']}

        This is a {row['FoodCategory'].lower()} food item that provides {row['Cals_per100grams']} calories per 100 grams.
        """.strip()

        # Rich metadata for filtering and exact lookups
        metadata = {
            "food_item": row["FoodItem"].lower(),
            "food_category": row["FoodCategory"].lower(),
            "calories_per_100g": (
                float(row["Cals_per100grams"])
                if pd.notna(row["Cals_per100grams"])
                else 0
            ),
            "kj_per_100g": (
                float(row["KJ_per100grams"]) if pd.notna(row["KJ_per100grams"]) else 0
            ),
            "serving_info": row["per100grams"],
            # Add searchable keywords
            "keywords": f"{row['FoodItem'].lower()} {row['FoodCategory'].lower()}".replace(
                " ", "_"
            ),
        }

        documents.append(document_text)
        metadatas.append(metadata)
        ids.append(f"food_{index}")

    return {"documents": documents, "metadatas": metadatas, "ids": ids}


def setup_nutrition_chromadb(csv_path: str, collection_name: str = "nutrition_db"):
    """
    Create and populate ChromaDB collection with nutrition data.
    """
    # Initialize ChromaDB
    client = chromadb.PersistentClient("./chroma")

    # Create collection (delete if exists)
    try:
        client.delete_collection(collection_name)
    except BaseException:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={
            "description": "Nutrition database with calorie and food information"
        },
    )

    # Prepare documents
    data = prepare_nutrition_documents(csv_path)

    # Add to ChromaDB
    collection.add(
        documents=data["documents"], metadatas=data["metadatas"], ids=data["ids"]
    )

    print(
        f"Added {len(data['documents'])} food items to ChromaDB collection '{collection_name}'"
    )
    return collection

if __name__ == "__main__":
    setup_nutrition_chromadb("calories.csv", "nutrition_db")
