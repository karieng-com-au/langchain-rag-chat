import random
import re
from pathlib import Path
from typing import Dict

import chromadb
from tqdm import tqdm


def prepare_questions_and_answers(
    qa_file: str,
    sample_percentage: float = 0.05
    ):
    """
    Create and populate ChromaDB collection with nutrition data.
    """
    try:
        qa_pairs = []
        with open(qa_file, "r", encoding="utf-8") as file:
            content = file.read()

        pairs = content.split("\n\n")

        total_pairs = len([p for p in pairs if p.strip()])
        sample_size = max(1, int(total_pairs * sample_percentage))

        print(f"Total Q&A pairs found: {total_pairs}")
        print(f"Sampling {sample_size} pairs ({sample_percentage*100:.1f}%)")

        valid_pairs = [p for p in pairs if p.strip()]

        sample_pairs = random.sample(valid_pairs, min(sample_size, len(valid_pairs)))

        for i, pair in enumerate(tqdm(sample_pairs, desc="Parsing Q&A pairs")):
            lines = pair.strip().split("\n")
            question = ""
            answer = ""

            for line in lines:
                if line.startswith("Question:"):
                    question = line.replace("Question:", "").strip()
                elif line.startswith("Answer:"):
                    answer = line.replace("Answer:", "").strip()
                if question and answer:
                    qa_pairs.append({"question": question, "answer": answer, "id": f"qa_{i}"})
    except FileNotFoundError as err:
        file.close()
        print(f"File is not found: {err}")
    except Exception as e:
        file.close()
        print(f"An error occurred: {e}")
    return qa_pairs

def prepare_nutrition_qa_documents(
    file_path: str, sample_percentage: float = 0.05
) -> Dict:
    """
    Convert Q&A pairs into ChromaDB-ready documents.
    Each Q&A pair becomes a searchable document.
    """
    qa_pairs = prepare_questions_and_answers(file_path, sample_percentage)

    documents = []
    metadatas = []
    ids = []

    # Process Q&A pairs with progress bar
    for qa in tqdm(qa_pairs, desc="Preparing documents"):
        # Create rich document text for semantic search
        document_text = f"""
        Question: {qa['question']}
        Answer: {qa['answer']}

        This Q&A pair provides information about nutrition and health topics.
        """.strip()

        # Extract keywords from question for better searchability
        question_words = re.findall(r"\b\w+\b", qa["question"].lower())
        answer_words = re.findall(r"\b\w+\b", qa["answer"].lower())
        all_words = question_words + answer_words

        # Create metadata for filtering and exact lookups
        metadata = {
            "question": qa["question"],
            "answer": qa["answer"],
            "question_length": len(qa["question"]),
            "answer_length": len(qa["answer"]),
            "keywords": " ".join(set(all_words)),
            "has_question_mark": "?" in qa["question"],
            "topic": "nutrition_qa",
        }

        documents.append(document_text)
        metadatas.append(metadata)
        ids.append(qa["id"])

    return {"documents": documents, "metadatas": metadatas, "ids": ids}

def setup_questions_and_answer_db(
    qa_file: str,
    database_path: str,
    collection_name: str = "qas_db",
    sample_percentage: float = 0.05
    ):
        # Initialize ChromaDB

        client = chromadb.PersistentClient(database_path)
        try:
            client.delete_collection(collection_name)
        except:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata={
                "description": "Nutrition Q&A database with questions and answers about nutrition and health"
            },
        )

        data = prepare_nutrition_qa_documents(qa_file, sample_percentage)

        # Add to ChromaDB
        collection.add(
            documents=data["documents"], metadatas=data["metadatas"], ids=data["ids"]
        )

        return collection

if __name__ == "__main__":
    collection = setup_questions_and_answer_db("./data/qna.txt", "./chroma", "qas_db", 0.05)
    print(f"Number of documents: {collection.count()}")
