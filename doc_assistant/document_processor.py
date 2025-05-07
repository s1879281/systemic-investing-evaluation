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
        """加载criteria文件"""
        criteria_path = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "input_files", "combined_hallmarks.json"))
        if not criteria_path.exists():
            raise FileNotFoundError("combined_hallmarks.json not found in input_files directory")
            
        with open(criteria_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def process_user_document(self, file_content, file_type):
        """处理用户上传的文档"""
        if file_type == 'txt':
            return file_content.decode('utf-8')
        elif file_type == 'docx':
            return self._process_docx(file_content)
        elif file_type == 'pdf':
            return self._process_pdf(file_content)
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    
    def _process_docx(self, file_content):
        """处理docx文件"""
        try:
            # 将文件内容转换为BytesIO对象
            docx_file = io.BytesIO(file_content)
            # 使用python-docx打开文档
            doc = Document(docx_file)
            # 提取所有段落的文本
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():  # 只添加非空段落
                    full_text.append(para.text)
            return '\n'.join(full_text)
        except Exception as e:
            raise ValueError(f"处理docx文件时出错: {str(e)}")
    
    def _process_pdf(self, file_content):
        """处理pdf文件，提取所有页面文本"""
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
            raise ValueError(f"处理pdf文件时出错: {str(e)}")
    
    def split_text(self, text, max_tokens=2000, model_name='gpt-4o'):
        """将长文本按最大token数分块"""
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
        """分块评估长文档，并聚合结果，递归摘要Justification/Indicators"""
        blocks = self.split_text(user_doc, max_tokens=max_tokens, model_name=model_name)
        chunk_results = []
        for i, block in enumerate(blocks):
            prompt = self.prepare_prompt(block)
            result = llm_service.get_evaluation(prompt)
            chunk_results.append(result)
            # 输出每个chunk的中间结果，便于debug
            st.subheader(f"[DEBUG] Chunk {i+1} 评估结果")
            st.write(result)
        # 聚合分数、justification/indicators
        from collections import defaultdict
        hallmark_scores = defaultdict(list)
        hallmark_justifications = defaultdict(list)
        hallmark_indicators = defaultdict(list)
        for chunk in chunk_results:
            table_md = chunk['table']
            # 解析表格，获取hallmark title顺序
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
        # 计算最大分，拼接justification/indicators
        final_scores = {h: max(v) for h, v in hallmark_scores.items() if v}
        final_justifications = {h: ' '.join(justs) for h, justs in hallmark_justifications.items()}
        final_indicators = {h: ' '.join(inds) for h, inds in hallmark_indicators.items()}
        # 递归摘要justification/indicators
        for h in final_justifications:
            prompt = f"请将以下关于 {h} 的评估理由进行总结归纳，输出一段简明扼要的说明：\n{final_justifications[h]}"
            summary = llm_service.get_evaluation(prompt)
            if isinstance(summary, dict) and 'table' in summary:
                final_justifications[h] = summary['table']
            else:
                final_justifications[h] = str(summary)
        for h in final_indicators:
            prompt = f"请将以下关于 {h} 的建议指标进行去重、归纳和总结，输出一段简明扼要的建议指标列表：\n{final_indicators[h]}"
            summary = llm_service.get_evaluation(prompt)
            if isinstance(summary, dict) and 'table' in summary:
                final_indicators[h] = summary['table']
            else:
                final_indicators[h] = str(summary)
        return final_scores, final_justifications, final_indicators

    def prepare_prompt(self, user_doc):
        """准备发送给LLM的prompt"""
        prompt = f"""Please evaluate the following case using the provided framework:

        Evaluation Framework (Hallmarks):
        {json.dumps(self.criteria, ensure_ascii=False, indent=2)}

        Case Document:
        {user_doc}
        """
        return prompt