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
    # Use a unique key for file_uploader to avoid duplicate key errors
    import uuid
    unique_key = f"category_upload_file_{uuid.uuid4()}"
    file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key=unique_key)
    if file:
        df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
        st.dataframe(df)
        if st.button("Save to DB"):
            # Drop 'S No' column if it exists
            if 'S No' in df.columns:
                df = df.drop(columns=['S No'])
            insert_data("category_data", df)
            st.success("Category data saved!")
            # Show all rows in category_data table after saving
            with get_db_connection() as conn:
                cat_df = pd.read_sql("SELECT * FROM category_data", conn)
            edited_df = st.data_editor(cat_df, num_rows="dynamic", use_container_width=True)
            # Optionally, you can add a button to save edits back to DB
            if st.button("Save Edits to DB", key="save_category_edits_btn"):
                # Overwrite the table with edited data
                with get_db_connection() as conn:
                    conn.execute("DELETE FROM category_data")
                    edited_df.to_sql("category_data", conn, if_exists="append", index=False)
                st.success("Edits saved to category_data table!")

# Ensure the UI is shown when the page loads
render_page()
