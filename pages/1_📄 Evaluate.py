import streamlit as st
import sys
import tiktoken
import os
import pickle
from datetime import datetime

def render_dataframe(df):
    st.markdown(
        '<style>table {word-break: break-word !important; white-space: pre-line !important; max-width: 1400px !important;} td {max-width: 600px; word-break: break-word !important; white-space: pre-line !important;}</style>' +
        df.to_html(index=False, escape=False),
        unsafe_allow_html=True
    )

def save_case_cache(case_name, score, table_html):
    cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'case_cache.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    else:
        cache = {}
    cache[case_name] = {
        'score': score,
        'table_html': table_html,
        'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

st.set_page_config(page_title="Case Evaluation", layout="wide")
st.title("Case Evaluation")

# Initialize session state variables
if 'initialized' not in st.session_state:
    try:
        from doc_assistant.document_processor import DocumentProcessor
        from doc_assistant.llm_service import LLMService, EvaluationVisualizer
        
        st.session_state.document_processor = DocumentProcessor()
        st.session_state.llm_service = LLMService()
        st.session_state.visualizer = EvaluationVisualizer()
        st.session_state.initialized = True
    except Exception as e:
        st.error(f"初始化服务时出错: {str(e)}")
        st.stop()

st.header("File Upload")
uploaded_files = st.file_uploader("Upload Case Documents", type=['txt', 'docx', 'pdf'], accept_multiple_files=True)
case_name = st.text_input("Enter a unique case name (used as identifier)")
evaluate_clicked = st.button("Evaluate")
if uploaded_files and case_name and evaluate_clicked:
    try:
        # 检查case名唯一性
        cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'case_cache.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
        else:
            cache = {}
        if case_name in cache:
            st.error("Case name already exists. Please enter a unique name.")
            st.stop()
        # 拼接所有文件内容
        processed_text = ""
        for uploaded_file in uploaded_files:
            file_type = uploaded_file.name.split('.')[-1].lower()
            file_content = uploaded_file.read()
            processed_text += st.session_state.document_processor.process_user_document(file_content, file_type) + "\n\n"
        st.session_state.current_doc = processed_text
        # 检查token数量
        enc = tiktoken.encoding_for_model('gpt-4o-mini')
        total_tokens = len(enc.encode(processed_text))
        max_tokens = 100_000
        if total_tokens > max_tokens:
            st.error(f"Document exceeds maximum token limit ({max_tokens:,} tokens). Current document has {total_tokens:,} tokens. Please upload a shorter document.")
            st.stop()
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        st.stop()
st.header("Evaluation Results")
if uploaded_files and case_name and evaluate_clicked and 'current_doc' in st.session_state:
    with st.spinner("Evaluating..."):
        try:
            # 短文档，单块评估
            prompt = st.session_state.document_processor.prepare_prompt(
                st.session_state.current_doc
            )
            result = st.session_state.llm_service.get_evaluation(prompt)
            if result:
                import pandas as pd
                import re
                table_md = result['table']
                # Split the table into lines and remove empty lines
                lines = [line.rstrip() for line in table_md.split('\n') if line.strip()]
                # Get header
                header_line = lines[0]
                header = [h.strip() for h in header_line.split('|')[1:-1]]
                # Skip the separator line (second line)
                data_lines = lines[2:]
                data = []
                buffer = []
                for line in data_lines:
                    if line.startswith('|'):
                        if buffer:
                            row_str = '\n'.join(buffer)
                            cols = [c.strip() for c in row_str.split('|')[1:-1]]
                            cols = [c.replace('\n', '; ') for c in cols]
                            if len(cols) < len(header):
                                cols += [''] * (len(header) - len(cols))
                            elif len(cols) > len(header):
                                cols = cols[:len(header)]
                            if any(c for c in cols):
                                data.append(cols)
                            buffer = []
                    buffer.append(line)
                if buffer:
                    row_str = '\n'.join(buffer)
                    cols = [c.strip() for c in row_str.split('|')[1:-1]]
                    cols = [c.replace('\n', '; ') for c in cols]
                    if len(cols) < len(header):
                        cols += [''] * (len(header) - len(cols))
                    elif len(cols) > len(header):
                        cols = cols[:len(header)]
                    if any(c for c in cols):
                        data.append(cols)
                df = pd.DataFrame(data, columns=header)
                df = df.dropna(how='all')
                table_html = df.to_html(index=False, escape=False)
                render_dataframe(df)
                st.subheader("Overall Score")
                st.write(f"Average Score: {result['overall_score']}")
                if isinstance(result['scores'], dict):
                    st.subheader("Score Distribution (Hallmarks)")
                    fig = st.session_state.visualizer.create_radar_chart(result['scores'], height=720, width=960)
                    st.plotly_chart(fig)
                    st.subheader("Score Distribution (Levels & Conditions)")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = st.session_state.visualizer.create_level_radar_chart(result['scores'], height=600, width=900)
                        st.plotly_chart(fig1)
                    with col2:
                        fig2 = st.session_state.visualizer.create_condition_radar_chart(result['scores'], height=600, width=900)
                        st.plotly_chart(fig2)
                else:
                    st.error("LLM returned scores in incorrect format. Please check LLM output format.")
                # 缓存case名、score和table_html
                save_case_cache(case_name, result['scores'], table_html)
            else:
                st.error("Unable to get evaluation results. Please try again.")
        except Exception as e:
            st.error(f"Error during evaluation: {str(e)}") 