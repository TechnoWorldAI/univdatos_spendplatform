"""Upload page module for data upload and validation."""
import streamlit as st
import pandas as pd
import io
from typing import Optional
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute
from .db import insert_data
import sqlite3
from datetime import datetime

def render_page() -> None:
    """Render the data upload page."""
    try:
        debug_logger.info("Rendering upload page")
        st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> </div>", unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a Spend file to upload",
            type=['xlsx', 'xls', 'csv'],
            help="Upload Excel or CSV files with spend transaction data"
        )
        
        if uploaded_file is not None:
            debug_logger.debug("File uploaded", {"filename": uploaded_file.name, "size": uploaded_file.size})
            
            success, df, error = safe_execute(
                pd.read_excel, uploaded_file,
                error_title="Failed to Read Excel File",
                show_ui_error=True
            )
            
            if not success or df is None:
                debug_logger.error("Failed to read uploaded file")
                return
                
            debug_logger.info("File read successfully", {"rows": len(df), "columns": len(df.columns)})
            st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
            
            # Preview data
           
            try:
                render_styled_table(df)
            except Exception:
                st.dataframe(df.head(10))
            
            # Validation
            debug_logger.debug("Starting data validation")
            success, validation_results, error = safe_execute(
                validate_data, df,
                error_title="Failed to Validate Data",
                show_ui_error=True
            )
            
            if success and validation_results:
                render_validation_results(validation_results)
                
                # Allow processing regardless of validation
                if st.button("🚀 Process Data"):
                    debug_logger.info("Processing data", {"rows": len(df)})
                    safe_execute(
                        process_data, df, validation_results,
                        error_title="Failed to Process Data",
                        show_ui_error=True
                    )
                        
    except Exception as e:
        debug_logger.exception("Error rendering upload page", e)
        show_error_block("Upload Page Error", e)


def validate_data(df: pd.DataFrame) -> dict:
    """Validate uploaded data and return validation results."""
    debug_logger.debug("Validating uploaded data", {"rows": len(df), "columns": len(df.columns)})
    
    results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'summary': {}
    }
    
    
    # Data quality checks
    # Negative value check
    negative_count = 0
    if 'total_amount' in df.columns:
        negative_values = df[df['total_amount'] < 0]
        negative_count = len(negative_values)
        if negative_count > 0:
            results['errors'].append(f"Found {negative_count} negative total_amount(s)")

    # Null value check
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        results['warnings'].append(f"Found {null_count} missing/null value(s) in the data")

    # Null supplier_name check
    missing_suppliers_count = 0
    if 'supplier_name' in df.columns:
        missing_suppliers = df[df['supplier_name'].isna()]
        missing_suppliers_count = len(missing_suppliers)
        if missing_suppliers_count > 0:
            results['warnings'].append(f"Found {missing_suppliers_count} records with missing supplier names")

    # Summary statistics
    results['summary'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'negative_values': negative_count,
        'missing_values': null_count,
        'missing_suppliers': missing_suppliers_count
    }

    return results


def render_validation_results(results: dict) -> None:
    """Render validation results."""
    st.markdown("<hr style='border:0.5px solid #D8D8D8; margin:0px;'>", unsafe_allow_html=True)
    #st.markdown("<div style='font-size:1.0em; color:#444; margin-bottom:18px;'>📋 Validation Results </div>", unsafe_allow_html=True)
    # Summary
    col1, col2, col3, col4, col5 = st.columns(5)
    card_style = "background-color:#f8f9fa; border-radius:10px; box-shadow:0 2px 8px #e0e0e0; padding:18px; margin-bottom:8px; text-align:center;"
    with col1:
        st.markdown(f"<div style='{card_style}'><span style='font-size:1.0em; color:#333;'>Total Rows</span><br><span style='font-size:1.3em; font-weight:bold; color:#007bff;'>{results['summary']['total_rows']}</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div style='{card_style}'><span style='font-size:1.0em; color:#333;'>Total Columns</span><br><span style='font-size:1.3em; font-weight:bold; color:#007bff;'>{results['summary']['total_columns']}</span></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div style='{card_style}'><span style='font-size:1.0em; color:#333;'>Negative Values</span><br><span style='font-size:1.3em; font-weight:bold; color:#dc3545;'>{results['summary']['negative_values']}</span></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div style='{card_style}'><span style='font-size:1.0em; color:#333;'>Missing Values</span><br><span style='font-size:1.3em; font-weight:bold; color:#ffc107;'>{results['summary']['missing_values']}</span></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div style='{card_style}'><span style='font-size:1.0em; color:#333;'>Missing Suppliers</span><br><span style='font-size:1.3em; font-weight:bold; color:#17a2b8;'>{results['summary']['missing_suppliers']}</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.0em; color:#444; margin-bottom:18px;'> </div>", unsafe_allow_html=True)
    st.markdown("<hr style='border:0.5px solid #D8D8D8; margin:0px;'>", unsafe_allow_html=True)
    # Display Error Management Tabs
    # Errors
    if results['errors']:
        st.error("❌ Validation Errors:")
        for error in results['errors']:
            st.error(f"• {error}")
    
    # Warnings
    if results['warnings']:
        st.warning("⚠️ Data Quality Warnings:\n" + "\n".join([f" {w}" for w in results['warnings']]))
    
    if results['is_valid'] and not results['warnings']:
        st.success("✅ All validations passed!")
    
    conn = sqlite3.connect('spend_platform.db')
    cursor = conn.cursor()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Insert errors
    for error in results.get('errors', []):
        cursor.execute("""
            INSERT INTO error_table (error_transaction, error_type, error_message, error_status, error_timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, ("Spend Upload", "Error", error, "Open", now_str))
    # Insert warnings
    for warning in results.get('warnings', []):
        cursor.execute("""
            INSERT INTO error_table (error_transaction, error_type, error_message, error_status, error_timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, ("Spend Upload", "Warning", warning, "Open", now_str))
    conn.commit()
    conn.close()
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
                ("width", "8%")
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

def process_data(df: pd.DataFrame, validation_results: dict = None) -> None:
    """Process and save validated data to database. Also log validation results to error_table."""
    insert_data("spend_data", df)
    # Insert validation errors/warnings into error_table
    st.success("Spend data saved!")
