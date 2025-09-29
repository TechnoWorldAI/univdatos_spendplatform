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
    st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'>Categorize Items</div>", unsafe_allow_html=True)

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
    render_styled_table(categorized_df)

    # Save edits to DB
    st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'></div>", unsafe_allow_html=True)
    if st.button("Save Categorized Data", key="save_categorized_data_btn_cat_edit"):
        insert_data("categorized_data", categorized_df)
        st.success("Categorized data saved!")


def render_styled_table(df):
    column_mapping = {
        "Item Description": "Item Description",
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
                ("width", "16.5%")
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
    scrollable_html = f'<div style="max-height:500px; overflow:auto; border:0px solid #ddd; padding:5px;">{html_table}</div>'

    st.markdown(scrollable_html, unsafe_allow_html=True)
# ✅ Run inside Streamlit page
#render_page()
