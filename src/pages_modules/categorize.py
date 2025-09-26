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
    edited_df = st.data_editor(
        categorized_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "item_description": st.column_config.TextColumn("Item Description", help="Description of the item"),
            "category_1": st.column_config.TextColumn("Category 1", help="Top-level category"),
            "category_2": st.column_config.TextColumn("Category 2", help="Second-level category"),
            "category_3": st.column_config.TextColumn("Category 3", help="Third-level category"),
            "category_4": st.column_config.TextColumn("Category 4", help="Fourth-level category"),
            "category_5": st.column_config.TextColumn("Category 5", help="Keywords or fifth-level category")
        }
    )

    # Save edits to DB
    if st.button("Save Categorized Data", key="save_categorized_data_btn_cat_edit"):
        insert_data("categorized_data", edited_df)
        st.success("Categorized data saved!")

# ✅ Run inside Streamlit page
#render_page()
