import streamlit as st
import os
import pickle
import pandas as pd

st.set_page_config(page_title="Manage Cases", layout="wide")
st.title("Manage Cases")

# 全局左对齐样式
st.markdown('''
    <style>
    .block-container {margin-left: 0 !important; padding-left: 1.5rem !important;}
    .stMarkdown, .stSelectbox, .stTextInput, .stButton {text-align: left !important;}
    .element-container {align-items: flex-start !important;}
    </style>
''', unsafe_allow_html=True)

cache_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cache', 'case_cache.pkl')
if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
else:
    cache = {}

# 记录上一次选择的case，切换时重置所有delete_mode
if 'last_selected_case' not in st.session_state:
    st.session_state['last_selected_case'] = None
if 'delete_success' not in st.session_state:
    st.session_state['delete_success'] = ""

st.markdown("<h1 style='text-align: left;'>Manage Evaluated Cases</h1>", unsafe_allow_html=True)

if not cache:
    st.info("No cached cases found.")
else:
    case_names = list(cache.keys())
    selected_case = st.selectbox("Select a case to manage", case_names)
    # 切换case时重置所有delete_mode
    if st.session_state['last_selected_case'] != selected_case:
        for k in list(st.session_state.keys()):
            if k.startswith('delete_mode_'):
                st.session_state[k] = False
        st.session_state['last_selected_case'] = selected_case
    # 删除成功提示
    if st.session_state['delete_success']:
        st.success(st.session_state['delete_success'])
        st.session_state['delete_success'] = ""
    if selected_case:
        st.write(f"**Current name:** {selected_case}")
        new_name = st.text_input("Enter new name to rename", value=selected_case, key="rename_input")
        col1, col2 = st.columns([2,2])
        with col1:
            if st.button("Rename"):
                if new_name != selected_case:
                    if new_name in cache:
                        st.error(f"Case name '{new_name}' already exists.")
                    else:
                        cache[new_name] = cache.pop(selected_case)
                        with open(cache_file, 'wb') as f:
                            pickle.dump(cache, f)
                        st.success(f"Renamed '{selected_case}' to '{new_name}'")
                        st.experimental_rerun()
        with col2:
            if f'delete_mode_{selected_case}' not in st.session_state:
                st.session_state[f'delete_mode_{selected_case}'] = False
            if st.button("Delete"):
                st.session_state[f'delete_mode_{selected_case}'] = True
                st.experimental_rerun()
            if st.session_state[f'delete_mode_{selected_case}']:
                confirm_col, cancel_col = st.columns([1,1], gap="small")
                with confirm_col:
                    if st.button("Confirm", key=f"confirm_{selected_case}"):
                        cache.pop(selected_case)
                        with open(cache_file, 'wb') as f:
                            pickle.dump(cache, f)
                        st.session_state['delete_success'] = f"Deleted '{selected_case}'"
                        st.session_state[f'delete_mode_{selected_case}'] = False
                        st.experimental_rerun()
                with cancel_col:
                    if st.button("Cancel", key=f"cancel_{selected_case}"):
                        st.session_state[f'delete_mode_{selected_case}'] = False
                        st.experimental_rerun()
        # Display the cached table for the selected case
        if selected_case in cache and 'table_html' in cache[selected_case]:
            st.subheader("Cached Table Preview")
            st.markdown(cache[selected_case]['table_html'], unsafe_allow_html=True) 