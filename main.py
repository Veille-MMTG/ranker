from pymongo import MongoClient
from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonformer.format import highlight_values
from jsonformer.main import Jsonformer
import json

import dotenv
import os
import time

dotenv.load_dotenv()

MONGO_URL = os.getenv("MONGO_URL")


def get_article():
    # Initialize MongoClient
    client = MongoClient(MONGO_URL)

    # Get the database and collection
    db = client["news_db"]
    collection = db["articles"]

    # Query articles without "quality" and "relevance" fields, limit to 1
    article_without_quality_relevance = collection.find_one(
        {"quality": {"$exists": False}, "relevance": {"$exists": False}}
    )

    return article_without_quality_relevance


def update_article_quality_relevance(article_id, quality, relevance):
    # Initialize MongoClient
    client = MongoClient(MONGO_URL)

    # Get the database and collection
    db = client["news_db"]
    collection = db["articles"]

    # Update article with quality and relevance
    collection.update_one(
        {"_id": article_id}, {"$set": {"quality": quality, "relevance": relevance}}
    )

    # check if article was updated
    article = collection.find_one({"_id": article_id})
    print("Updated article: " + article["title"])
    print("Quality: " + str(article["quality"]))
    print("Relevance: " + str(article["relevance"]))
    print("")


def rank_articles():
    print("Loading model and tokenizer...")
    
    model_name = "databricks/dolly-v2-3b"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, use_cache=True, device_map="balanced"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_cache=True)
    print("Loaded model and tokenizer")

    jsonFormat = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "quality": {"type": "number"},
            "relevance": {"type": "number"},
        },
        "required": ["title", "quality", "relevance"],
    }
    
    while True:
        # Get article from MongoDB
        article = get_article()
        if not article:
            break

        article_id = article["_id"]
        article_title = article["title"]

        prompt = f"As an AI language model, evaluate the following article titled '{article_title}' based on its quality and relevance to the topic of Large Language Models. Consider the following criteria when providing scores: clarity, specificity, novelty, and depth. Provide a quality score and a relevance score for this article, where both scores range from 0 to 10, with 10 being the highest."
        print("Prompt: " + prompt)

        print("Building Jsonformer...")
        builder = Jsonformer(
            model=model,
            tokenizer=tokenizer,
            json_schema=jsonFormat,
            prompt=prompt,
            max_string_token_length=20,
        )

        print("Generating...")
        output = builder()

        # Extract quality and relevance from the output
        article_quality = output["quality"]
        article_relevance = output["relevance"]

        # Update article in MongoDB with quality and relevance
        update_article_quality_relevance(article_id, article_quality, article_relevance)


def run():
    rank_articles()


if __name__ == "__main__":
    rank_articles()
