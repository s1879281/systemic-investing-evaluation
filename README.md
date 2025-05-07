# Document Intelligence Assistant MVP

This is a implementation of a document intelligence assistant system based on LLMs. The system supports uploading text documents and can automatically perform systemic investing evaluation based on the document content.

## Requirements

- Python 3.8+
- OpenAI API key

## Installation Steps

1. Clone the project and enter the project directory:
```bash
git clone <repository-url>
cd doc_assistant
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run Home.py
```

The app will start at http://localhost:8501

## Usage Instructions

1. Open your browser and visit the app
2. Upload a case document in `.txt` or `.docx` format
3. The system will automatically evaluate and display a detailed table, overall score, and radar chart

## Notes

- Make sure you have a valid OpenAI API key before use
- Be aware of API usage costs
- Processing large documents may take longer
- It is recommended to test in a development environment before deploying to production
- For caching, you can restore the `@st.cache_data` decorator in `llm_service.py`

## Deployment

For deploying to Streamlit Cloud:
1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. Add your OpenAI API key in the Streamlit Cloud dashboard under Settings → Secrets

## Future Improvements

- Add persistent storage
- Implement user authentication
- Improve user interface