# IngredientIQ — VS Code RAG Project

## Files
- `streamlit_app.py`: Streamlit chat interface.
- `retriever.py`: local sentence embeddings and semantic retrieval.
- `product_matcher.py`: product name matching and intent routing.
- `llm.py`: Groq response generation.
- `nutrition.py`: exact nutrition display, additive categories, comparison.
- `data_loader.py`: reads and cleans CSVs.
- `data/products.csv`: 70-product dataset.
- `data/additives.csv`: additive reference dataset.

## Recommended environment
Python 3.12 is recommended for dependency compatibility. Install Python 3.12 alongside any existing Python version if needed.

## Setup in PowerShell
From the project folder:
```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```
Add your Groq API key to `.env`, save, and run:
```powershell
python -m streamlit run streamlit_app.py
```

The first run downloads `all-MiniLM-L6-v2` (a small embedding model). It needs an internet connection for the first download; afterward, the model is cached locally.

Do not share `.env` or commit it to GitHub. Answers are grounded in the supplied CSVs; “not listed” does not prove an ingredient is absent.
