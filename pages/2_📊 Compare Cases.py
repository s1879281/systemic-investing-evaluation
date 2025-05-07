import streamlit as st
import os
import pickle
import pandas as pd
import json
import plotly.graph_objects as go

st.set_page_config(page_title="Score Comparison", layout="wide")
st.title("Compare Cases")

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

# 强制所有内容左对齐
st.markdown('''
    <style>
    .block-container {margin-left: 0 !important; padding-left: 1.5rem !important;}
    .stDataFrame, .stMarkdown, .stMultiSelect, .stSelectbox, .stTextInput, .stButton {
        text-align: left !important;
        justify-content: flex-start !important;
        align-items: flex-start !important;
    }
    .element-container {align-items: flex-start !important;}
    </style>
''', unsafe_allow_html=True)

cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'case_cache.pkl')
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
else:
    cache = {}

# 更学术的标题
st.markdown("<h1 style='text-align: left;'>Systemic Investing Hallmark Score Comparison</h1>", unsafe_allow_html=True)

# 读取分组mapping
mapping_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'input_files', 'system_change_level_to_hallmarks.json'))
if os.path.exists(mapping_path):
    with open(mapping_path, 'r', encoding='utf-8') as f:
        group_map = json.load(f)
else:
    group_map = {}

# 反向映射：hallmark -> group
hallmark_to_group = {}
for group, hallmarks in group_map.items():
    for h in hallmarks:
        hallmark_to_group[h] = group

# 颜色映射
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex

group_cmap = {
    "Transformational Change (implicit)": "YlOrBr",
    "Relational Change(semi-explicit)": "PuBu",
    "Structural Change(explicit)": "BuGn"
}

def multi_group_color(val, col, vmin, vmax, cmap_name):
    if pd.isna(val):
        return ''
    try:
        v = float(val)
    except:
        return ''
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)
    color = to_hex(cmap(norm(v)))
    return f'background-color: {color};'

if not cache:
    st.info("No cached cases found.")
else:
    case_names = list(cache.keys())
    selected = st.multiselect("Select cases to compare", case_names)
    if selected:
        scores = {name: cache[name]['score'] for name in selected}
        df = pd.DataFrame(scores).T
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.round(1)
        # 分组着色，vmin=0, vmax=20
        def style_func(row):
            styles = []
            for col, val in row.items():
                group = hallmark_to_group.get(col, None)
                cmap_name = group_cmap.get(group, "Greys")
                vmin, vmax = 0, 20
                styles.append(multi_group_color(val, col, vmin, vmax, cmap_name))
            return styles
        styled_df = df.style.apply(style_func, axis=1).format('{:.1f}')
        # 自定义表头样式，固定宽度并自动换行
        st.markdown(
            '''<style>
            th {
                max-width: 120px;
                min-width: 80px;
                word-break: break-word !important;
                white-space: pre-line !important;
                text-align: left !important;
            }
            td {
                text-align: left !important;
            }
            table {width: 100% !important;}
            </style>''', unsafe_allow_html=True
        )
        st.markdown(styled_df.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.info("Please select at least one case.") 