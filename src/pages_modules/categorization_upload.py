
"""Categorization upload and import page.

Allows uploading a categorization CSV/XLSX matching the context/Categorization File.csv
and importing hierarchical categories into the database.
"""
import streamlit as st
import pandas as pd
import os
from typing import Optional, Tuple, List
from src.pages_modules.db import insert_data
from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block
from src.utils.display import normalize_df_for_display

def render_page():
    st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> 📥 Upload Categorization File</div>", unsafe_allow_html=True)
    file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], help="Upload Excel or CSV files with spend transaction data")
    if file:
        try:
            df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        if st.button("Save to DB"):
            if 'S No' in df.columns:
                df = df.drop(columns=['S No'])
            insert_data("category_data", df)
            st.success("Category data saved!")
            with get_db_connection() as conn:
                cat_df = pd.read_sql("SELECT * FROM category_data", conn)
            edited_df = st.data_editor(cat_df, num_rows="dynamic", use_container_width=True)
            if st.button("Save Edits to DB", key="save_category_edits_btn"):
                with get_db_connection() as conn:
                    conn.execute("DELETE FROM category_data")
                    edited_df.to_sql("category_data", conn, if_exists="append", index=False)
                st.success("Edits saved to category_data table!")

# Do NOT call render_page() here.
# The main app should call render_page() only if the user is authenticated:
# Example usage in app.py:
# if st.session_state.get('authenticated', False):
#     from src.pages_modules.categorization_upload import render_page
#     render_page()


