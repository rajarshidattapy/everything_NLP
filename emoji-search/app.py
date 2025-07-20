import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch

from search.utils import load_emoji_dataset, get_base_text, enrich_with_gemini

st.set_page_config(page_title="Emoji Search with AI", page_icon="🔍")

st.title("🔍 Emoji Search Engine")
st.markdown("Search emojis using natural language — powered by sentence-transformers + Gemini-Pro")

query = st.text_input("Type your search (e.g., 'namaste', 'feeling strong', 'laughing till I cry')")

if query:
    with st.spinner("Thinking..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emoji_data = load_emoji_dataset()

        query_embedding = model.encode(query, convert_to_tensor=True)
        emoji_texts = []

        for entry in emoji_data:
            base_text = get_base_text(entry)
            enriched = enrich_with_gemini(entry, query)
            full_text = f"{base_text} {' '.join(enriched)}"
            emoji_texts.append(full_text)

        emoji_embeddings = model.encode(emoji_texts, convert_to_tensor=True)
        similarities = util.cos_sim(query_embedding, emoji_embeddings)[0]
        top_indices = torch.topk(similarities, k=5).indices

        st.markdown("### 🔎 Top Matching Emojis:")
        result_emojis = [emoji_data[i]["emoji"] for i in top_indices]
        st.write(" ".join(result_emojis))
