
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
            #st.dataframe(df)
            render_styled_table(df)
            st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'>  </div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
        if st.button("Save Category"):
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

def render_styled_table(df):


    styled_df = (
        df.style
        .hide(axis="index")
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1D2951"),
                ("color", "white"),
                ("text-align", "center"),
                ("border", "1px solid black"),
                ("padding", "1px"),
                ("font-size", "0.85em"),
                ("width", "20%")
            ]},
            {"selector": "td", "props": [
                ("border", "1px solid black"),
                ("text-align", "center"),
                ("padding", "4px"),
                ("font-size", "0.8em")
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "collapse"),
                ("border-radius", "6px"),
                ("width", "100%")
            ]}
        ], overwrite=False)  # 👈 ensure it merges styles properly
    )

    # Wrap with scrollable container
    html_table = styled_df.to_html(escape=False, table_attributes='class="dataframe"')

    # Wrap in scrollable container
    scrollable_html = f'<div style="max-height:400px; overflow:auto; border:0px solid #ddd; padding:5px;">{html_table}</div>'

    st.markdown(scrollable_html, unsafe_allow_html=True)



