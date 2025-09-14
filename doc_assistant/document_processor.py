import json
from pathlib import Path
from docx import Document
import io
import pdfplumber
import tiktoken
import streamlit as st
import os

class DocumentProcessor:
    def __init__(self):
        self.criteria = self._load_criteria()
        
    def _load_criteria(self):
        """Load criteria file"""
        criteria_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "input_files", "combined_hallmarks.json"))
        if not criteria_path.exists():
            raise FileNotFoundError("combined_hallmarks.json not found in input_files directory")
            
        with open(criteria_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_user_document(self, file_content, file_type):
        """Process user uploaded document"""
        if file_type == 'txt':
            return file_content.decode('utf-8')
        elif file_type == 'docx':
            return self._process_docx(file_content)
        elif file_type == 'pdf':
            return self._process_pdf(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _process_docx(self, file_content):
        """Process docx file"""
        try:
            # Convert file content to BytesIO object
            docx_file = io.BytesIO(file_content)
            # Use python-docx to open document
            doc = Document(docx_file)
            # Extract text from all paragraphs
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():  # Only add non-empty paragraphs
                    full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            raise ValueError(f"Error processing docx file: {str(e)}")
    
    def _process_pdf(self, file_content):
        """Process pdf file, extract all page text"""
        try:
            pdf_file = io.BytesIO(file_content)
            text = []
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return '\n'.join(text)
        except Exception as e:
            raise ValueError(f"Error processing pdf file: {str(e)}")
    
    def split_text(self, text, max_tokens=2000, model_name='gpt-4o'):
        """Split long text into chunks by maximum token number"""
        enc = tiktoken.encoding_for_model(model_name)
        lines = text.split('\n')
        blocks = []
        current = []
        token_count = 0
        for line in lines:
            line_tokens = len(enc.encode(line))
            if token_count + line_tokens > max_tokens and current:
                blocks.append('\n'.join(current))
                current = []
                token_count = 0
            current.append(line)
            token_count += line_tokens
        if current:
            blocks.append('\n'.join(current))
        return blocks

    def process_long_document(self, user_doc, llm_service, max_tokens=2000, model_name='gpt-4o'):
        """Chunk-wise evaluation of long documents, aggregate results, and recursively summarize Justification/Indicators"""
        blocks = self.split_text(user_doc, max_tokens=max_tokens, model_name=model_name)
        chunk_results = []
        for i, block in enumerate(blocks):
            prompt = self.prepare_prompt(block)
            result = llm_service.get_evaluation(prompt)
            chunk_results.append(result)
            # Output intermediate results for each chunk, for debugging
            st.subheader(f"[DEBUG] Chunk {i+1} Evaluation Result")
            st.write(result)
        # Aggregate scores, justification/indicators
        from collections import defaultdict
        hallmark_scores = defaultdict(list)
        hallmark_justifications = defaultdict(list)
        hallmark_indicators = defaultdict(list)
        for chunk in chunk_results:
            table_md = chunk['table']
            # Parse table, get hallmark title order
            lines = [line for line in table_md.split('\n') if '|' in line and not line.strip().startswith('|--')]
            if lines:
                header = [h.strip() for h in lines[0].split('|')[1:-1]]
                for row in lines[1:]:
                    cols = [c.strip() for c in row.split('|')[1:-1]]
                    if len(cols) >= 4:
                        hallmark = cols[0]
                        score = float(cols[1])
                        justification = cols[2]
                        indicators = cols[3]
                        hallmark_scores[hallmark].append(score)
                        hallmark_justifications[hallmark].append(justification)
                        hallmark_indicators[hallmark].append(indicators)
        # Calculate maximum score, concatenate justification/indicators
        final_scores = {h: max(v) for h, v in hallmark_scores.items() if v}
        final_justifications = {h: ' '.join(justs) for h, justs in hallmark_justifications.items()}
        final_indicators = {h: ' '.join(inds) for h, inds in hallmark_indicators.items()}
        # Recursively summarize justification/indicators
        for h in final_justifications:
            prompt = f"Please summarize the evaluation reasons for {h} in a concise manner: \n{final_justifications[h]}"
            summary = llm_service.get_evaluation(prompt)
            if isinstance(summary, dict) and 'table' in summary:
                final_justifications[h] = summary['table']
            else:
                final_justifications[h] = str(summary)
        for h in final_indicators:
            prompt = f"Please summarize the suggested indicators for {h} in a concise manner: \n{final_indicators[h]}"
            summary = llm_service.get_evaluation(prompt)
            if isinstance(summary, dict) and 'table' in summary:
                final_indicators[h] = summary['table']
            else:
                final_indicators[h] = str(summary)
        return final_scores, final_justifications, final_indicators

    def prepare_prompt(self, user_doc):
        """Prepare prompt to send to LLM"""
        prompt = f"""Please evaluate the following case using the provided framework:

        Evaluation Framework (Hallmarks):
        {json.dumps(self.criteria, ensure_ascii=False, indent=2)}

        Case Document:
        {user_doc}
        """
        return prompt
