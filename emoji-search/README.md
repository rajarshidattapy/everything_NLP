
# 🤖 Emoji NLP Search 🔍

Search emojis using natural language — powered by **Gemini-Pro** and **sentence-transformers**.

## 🚀 Features

- 🔍 Semantic emoji search (e.g., "namaste", "feeling happy", "broken heart")
- 🤖 Real-time tag enrichment using Google Gemini-Pro
- ⚡ Fast vector similarity using `sentence-transformers`
- 🖥️ Simple UI with Streamlit

## 🛠 Tech Stack

- Gemini-Pro (`google-generativeai`)
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Streamlit UI
- PyTorch for embeddings

## 📂 Folder Structure

```

emoji-nlp-search/
├── data/                       # Raw emoji JSON dataset
├── search/                    # Search logic + Gemini helpers
├── app/                       # Streamlit frontend
├── .env                       # Gemini API Key
├── requirements.txt
└── README.md

````

## ✅ Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
````

2. **Add your Gemini API key to `.env`**

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Run the app**

   ```bash
   streamlit run app/app.py
   ```

## 📘 Example Searches

* `namaste` → 🙏
* `birthday celebration` → 🎂🎉
* `feeling sad` → 😢💔
* `laughing hard` → 😂🤣

---

Built with ❤️ by Rajarshi

```

