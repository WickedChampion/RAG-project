# RAG-practical — Streamlit + LangChain + Supabase

Quick demo project to upload documents, index into Supabase (vector) and query with LangChain.

Setup

1. Create a Python venv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your Supabase and OpenAI keys:

```text
SUPABASE_URL=your-supabase-url
SUPABASE_KEY=your-service-key-or-anon-key
OPENAI_API_KEY=your-openai-key-optional
```

3. Run the app:

```bash
streamlit run app.py
```

Notes
- The app sends a magic link to the email you enter in the sidebar (Supabase Auth).
- Uploaded PDFs/TXT are chunked and indexed into a Supabase vector table named `documents` using LangChain's SupabaseVectorStore.
- If `OPENAI_API_KEY` is set the app will use OpenAI to generate answers; otherwise it will show retrieved chunks.

If you want, I can run the app locally here and verify the flow.
