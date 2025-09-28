import streamlit as st
import pandas as pd
import sqlite3
import io
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute
from .db import DB_NAME, insert_data


def render_page():
    #st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> 🔧 Error Management </div>", unsafe_allow_html=True)
    # Show all rows in error_table and make editable
    error_df = load_errors()
   
    # Display Error Management Tabs
    render_data_errors()

    st.markdown("<hr style='border:0.5px solid #D8D8D8; margin:0px;'>", unsafe_allow_html=True)
    display_error_table(error_df)

def display_error_table(error_df):
    st.markdown("<div style='font-size:1.0em; color:#444; margin-bottom:18px;'> Existing Resolved Errors </div>", unsafe_allow_html=True)
    resolved_errors_df = error_df[error_df['error_status'] == 'Resolved']
    render_styled_table(resolved_errors_df)

def render_data_errors():
    """Render data errors section."""
    
    #st.markdown(f"<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> Data Processing Errors </div>", unsafe_allow_html=True)

    # Load error data
    success, errors_df, error = safe_execute(
        load_errors,
        error_title="Failed to Load Errors",
        show_ui_error=True
    )
    
    if not success or errors_df is None:
        return

    
    # Summary metrics in card format
    col1, col2, col3 = st.columns(3)
    card_style = "background-color:#f8f9fa; border-radius:6px; box-shadow:0 1px 4px grey; padding:0px; margin-bottom:16px; text-align:center;"
    with col1:
        st.markdown(f"<div style='{card_style}'><span style='font-size:0.95em; color:#333;'>Total Errors</span><br><span style='font-size:1.15em; font-weight:bold; color:#007bff;'>{len(errors_df)}</span></div>", unsafe_allow_html=True)
    with col2:
        open_errors = errors_df[errors_df['error_status'] == 'Open']
        st.markdown(f"<div style='{card_style}'><span style='font-size:0.95em; color:#333;'>Open Errors</span><br><span style='font-size:1.15em; font-weight:bold; color:#dc3545;'>{len(open_errors)}</span></div>", unsafe_allow_html=True)
    with col3:
        resolved_errors = errors_df[errors_df['error_status'] == 'Resolved']
        st.markdown(f"<div style='{card_style}'><span style='font-size:0.95em; color:#333;'>Resolved Errors</span><br><span style='font-size:1.15em; font-weight:bold; color:green;'>{len(resolved_errors)}</span></div>", unsafe_allow_html=True)

    # Display errors
    #st.markdown("<hr style='border:0.5px solid #D8D8D8; margin-top:10px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.0em; color:#444; margin-top:10px;margin-bottom: 18px'> Resolve Data Processing Errors </div>", unsafe_allow_html=True)
    open_errors_df = errors_df[errors_df['error_status'] == 'Open']
    if open_errors_df.empty:
        st.info("No data processing errors found.")

    if not open_errors_df.empty:
        # Display error table first
        render_styled_table(open_errors_df)
        # Action button below table
        if st.button("Resolve Errors"):
            resolve_all_errors(open_errors_df['error_id'].tolist())
            st.success("All open errors marked as resolved!")
            st.rerun()
            st.info("No open data processing errors found.")

def resolve_all_errors(error_ids: list) -> None:
    """Resolve multiple errors."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        for error_id in error_ids:
            cursor.execute("""
                UPDATE error_table
                SET error_status = 'Resolved',
                    resolved_on = datetime('now'),
                    is_resolved = ?
                WHERE error_id = ?
            """, (1, error_id))

        conn.commit()

def render_styled_table(df):
    column_mapping = {
        "error_id": "Error ID",
        "error_transaction": "Error Transaction",
        "error_type": "Error Type",
        "error_message": "Error Message",
        "error_status": "Error Status",
        "error_timestamp": "Error Created",
        "is_resolved": "Is Resolved",
        "resolved_on": "Resolved On"
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
                ("font-size", "0.85em")
                #("width", "12.5%")
            ]},
            {"selector": "td", "props": [
                ("border", "1px solid black"),
                ("text-align", "center"),
                ("padding", "4px"),
                ("font-size", "0.8em")
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "expand"),
                ("border-radius", "6px"),
                ("width", "100%")
            ]}
        ])
    )
    html_table = styled_df.to_html(escape=False, table_attributes='class="dataframe"')

    # Wrap in scrollable container
    scrollable_html = f'<div style="max-height:300px; overflow:auto; border:0px solid #ddd; padding:5px;">{html_table}</div>'

    st.markdown(scrollable_html, unsafe_allow_html=True)


def load_errors():
    """Load all rows from error_table as a DataFrame."""
    with get_db_connection() as conn:
        error_df = pd.read_sql("SELECT * FROM error_table", conn)
    return error_df
