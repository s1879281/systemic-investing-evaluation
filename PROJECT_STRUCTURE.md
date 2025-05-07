# Project Structure

This document describes the recommended directory and file layout for the Streamlit-based Document Intelligence Assistant project.

```
your_project_root/
│
├── Home.py                       # Streamlit main entry (welcome page)
├── requirements.txt              # Python dependencies
├── .env                          # (local, not committed) OpenAI API key, etc.
├── .gitignore
├── .cursorignore
├── README.md                     # Developer documentation
│
├── doc_assistant/                # Core logic and service modules
│   ├── __init__.py
│   ├── document_processor.py     # Document parsing and processing
│   ├── llm_service.py            # LLM API and visualization logic
│
├── pages/                        # Streamlit multi-page app scripts
│   ├── 1_📄 Evaluate.py          # Case evaluation page
│   ├── 2_📊 Compare Cases.py     # Case comparison page
│   ├── 3_⚙️ Manage Cases.py      # Case management page
│
├── input_files/                  # Input configuration files (JSON, etc.)
│   └── (your .json files)
│
├── cache/                        # Cached evaluation results
│   └── case_cache.pkl
│
```

## Directory & File Explanations

- **Home.py**: Main Streamlit entry point (welcome/landing page).
- **requirements.txt**: All Python dependencies for the project.
- **.env / .env.example**: Environment variables, e.g., OpenAI API key.
- **doc_assistant/**: Core backend logic, document processing, LLM service, and visualization.
- **pages/**: Streamlit multi-page app scripts, each file is a separate page.
- **input_files/**: Place for all input configuration files (e.g., hallmark mappings, settings, etc.).
- **cache/**: Stores cached evaluation results (e.g., case_cache.pkl). Should be writable by the app.

## Notes
- Make sure `input_files/`, `cache/` directories exist before running the app.
- Do not commit sensitive files (like `.env`) to version control.
- You can add or reorganize utility scripts as needed for your workflow. 