import json
import torch
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/emoji_dataset_raw.json", "r", encoding="utf-8") as f:
    emoji_data = json.load(f)

# Combine basic text info for each emoji
def get_emoji_text(entry):
    return f"{entry['description']} {' '.join(entry.get('aliases', []))} {' '.join(entry.get('tags', []))}"

# Optional: Gemini enrichment per emoji
def enrich_with_gemini(entry, query):
    prompt = f"""
You're helping match emoji to text.

Given:
- Query: "{query}"
- Emoji: {entry['emoji']} ({entry['description']})
- Category: {entry['category']}
- Aliases: {entry.get('aliases', [])}
- Tags: {entry.get('tags', [])}

Suggest some short phrases or synonyms people might use to refer to this emoji in this context. Return as a list.
"""
    try:
        response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
        additions = json.loads(response.text)
        return additions if isinstance(additions, list) else []
    except Exception as e:
        print(f"Gemini failed on {entry['emoji']}: {e}")
        return []

# Main search function
def search_emojis(query, top_k=5, use_gemini=True):
    query_embedding = model.encode(query, convert_to_tensor=True)
    emoji_texts = []
    
    for entry in emoji_data:
        base_text = get_emoji_text(entry)
        if use_gemini:
            enriched = enrich_with_gemini(entry, query)
            full_text = f"{base_text} {' '.join(enriched)}"
        else:
            full_text = base_text
        emoji_texts.append(full_text)

    emoji_embeddings = model.encode(emoji_texts, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, emoji_embeddings)[0]
    top_indices = torch.topk(similarities, k=top_k).indices

    return [emoji_data[i]["emoji"] for i in top_indices]
