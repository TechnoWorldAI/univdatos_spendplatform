import streamlit as st
import pandas as pd
import sqlite3
import io
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .db import DB_NAME, insert_data

def render_page():
    st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'>📂 Categorize Items</div>", unsafe_allow_html=True)

    conn = sqlite3.connect(DB_NAME)
    spend_df = pd.read_sql("SELECT item_description FROM spend_data", conn)
    cat_df = pd.read_sql("SELECT * FROM category_data", conn)
    conn.close()

    if spend_df.empty or cat_df.empty:
        st.warning("Upload spend and category data first")
        return

    # ✅ Use Category 5 as keywords
    if "Category 5" not in cat_df.columns:
        st.warning("'Category 5' column not found in category data. Please upload with Category 5 as keywords.")
        return

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cat_df["Category 5"].fillna(""))

    results = []
    for _, row in spend_df.iterrows():
        item_desc = str(row["item_description"])

        # Transform item description
        item_vec = vectorizer.transform([item_desc])
        sims = cosine_similarity(item_vec, tfidf_matrix)
        best_idx = sims.argmax()

        # Extract category levels
        cat_levels = []
        for col in ["Category 1", "Category 2", "Category 3", "Category 4", "Category 5"]:
            if col in cat_df.columns:
                val = cat_df.loc[best_idx, col]
                cat_levels.append(str(val) if pd.notna(val) else "")
            else:
                cat_levels.append("")

        confidence = sims[0, best_idx]
        results.append([item_desc] + cat_levels)

    categorized_df = pd.DataFrame(results, columns=["item_description","category_1","category_2","category_3","category_4","category_5"])
    edited_df = st.data_editor(categorized_df, num_rows="dynamic", use_container_width=True)

    # Save edits to DB
    if st.button("Save Categorized Data", key="save_categorized_data_btn_cat_edit"):
        insert_data("categorized_data", edited_df)
        st.success("Categorized data saved!")

    # Delete selected rows
    selected_rows = st.multiselect("Select rows to delete", edited_df.index.tolist(), key="delete_categorized_rows")
    if st.button("Delete Selected Rows", key="delete_categorized_data_btn_cat"):
        df_after_delete = edited_df.drop(index=selected_rows)
        # Overwrite the table with remaining data
        conn = sqlite3.connect(DB_NAME)
        conn.execute("DELETE FROM categorized_data")
        df_after_delete.to_sql("categorized_data", conn, if_exists="append", index=False)
        conn.close()
        st.success("Selected rows deleted from categorized_data table!")

# ✅ Run inside Streamlit page
render_page()
