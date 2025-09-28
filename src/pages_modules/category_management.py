"""Category management page module."""
import streamlit as st
import pandas as pd
import os
from typing import Optional, Tuple, List
from src.pages_modules.db import insert_data
from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block
from src.utils.display import normalize_df_for_display


def render_page():
    st.markdown("<div style='font-size:1.0em; color:#444; margin-bottom:18px;'> Existing Category Data Table</div>", unsafe_allow_html=True)
    # Show all rows in category_data table and make editable
    with get_db_connection() as conn:
        cat_df = pd.read_sql("SELECT * FROM category_data", conn)
    import uuid
    data_editor_key = f"category_data_editor_{uuid.uuid4()}"
    render_styled_table(cat_df)

def render_styled_table(df):
    column_mapping = {
        "Category 1": "Category 1",
        "Category 2": "Category 2",
        "Category 3": "Category 3",
        "Category 4": "Category 4",
        "Category 5": "Category 5"
    }

    df = df.rename(columns=column_mapping)

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
        ], overwrite=False)
    )

    # Convert to HTML without outer <div> (table only)
    html_table = styled_df.to_html(escape=False, table_attributes='class="dataframe"')

    # Wrap in scrollable container
    scrollable_html = f'<div style="max-height:300px; overflow:auto; border:0px solid #ddd; padding:5px;">{html_table}</div>'

    st.markdown(scrollable_html, unsafe_allow_html=True)

# Ensure the UI is shown when the page loads
