import json
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load emoji data from JSON
def load_emoji_dataset(path="data/emoji_dataset_raw.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Combine emoji description, aliases, and tags into one string
def get_base_text(entry):
    desc = entry.get("description", "")
    aliases = " ".join(entry.get("aliases", []))
    tags = " ".join(entry.get("tags", []))
    return f"{desc} {aliases} {tags}"

# Use Gemini to generate real-time synonyms/context words
def enrich_with_gemini(entry, query):
    prompt = f"""
You're an emoji matcher. The user typed:

Query: "{query}"

This emoji is:
Emoji: {entry['emoji']}
Description: {entry['description']}
Category: {entry['category']}
Aliases: {entry.get('aliases', [])}
Tags: {entry.get('tags', [])}

Suggest a few short search terms (synonyms or phrases) that people might type when looking for this emoji **in this context**.
Respond in a Python list like this: ["term1", "term2", "term3"]
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        additions = json.loads(response.text)
        return additions if isinstance(additions, list) else []
    except Exception as e:
        print(f"[Gemini Error] {entry['emoji']}: {e}")
        return []
