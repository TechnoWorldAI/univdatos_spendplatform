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
    st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> 📂 Existing Category Data Table</div>", unsafe_allow_html=True)
    # Show all rows in category_data table and make editable
    with get_db_connection() as conn:
        cat_df = pd.read_sql("SELECT * FROM category_data", conn)
    import uuid
    data_editor_key = f"category_data_editor_{uuid.uuid4()}"
    edited_df = st.data_editor(cat_df, num_rows="dynamic", use_container_width=True, key=data_editor_key)
    if st.button("Save Changes", key="save_category_edits_btn"):
        with get_db_connection() as conn:
            conn.execute("DELETE FROM category_data")
            edited_df.to_sql("category_data", conn, if_exists="append", index=False)
        st.success("Edits saved to category_data table!")

# Ensure the UI is shown when the page loads
render_page()