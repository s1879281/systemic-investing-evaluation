import streamlit as st
st.set_page_config(page_title="Welcome", layout="wide")
st.title("Systemic Investing Assistant")
st.markdown("""
Welcome to the Systemic Investing Assistant!

- **Evaluate**: Upload and evaluate a single case to obtain systemic investing scores and visualizations.
- **Compare Cases**: Compare hallmark scores across multiple cases in batch.
- **Manage Cases**: Manage cached cases, including renaming and deletion.

Please use the left sidebar to select a function page.
""") 