"""Upload page module for data upload and validation."""
import streamlit as st
import pandas as pd
import io
from typing import Optional
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute
from .db import insert_data


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
            with st.expander("📋 Data Preview"):
                try:
                    from src.utils.display import normalize_df_for_display
                    st.dataframe(normalize_df_for_display(df.head(10)))
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
                        process_data, df,
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
    if 'item_invoice_value' in df.columns:
        negative_values = df[df['item_invoice_value'] < 0]
        negative_count = len(negative_values)
        if negative_count > 0:
            results['errors'].append(f"Found {negative_count} negative item_invoice_value(s)")

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
        'missing_values': null_count
    }

    return results


def render_validation_results(results: dict) -> None:
    """Render validation results."""

    st.markdown("<div style='font-size:1.0em; color:#444; margin-bottom:18px;'>📋 Validation Results </div>", unsafe_allow_html=True)
    # Summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", results['summary']['total_rows'])
    with col2:
        st.metric("Total Columns", results['summary']['total_columns'])
    with col3:
        st.metric("Negative Values", results['summary']['negative_values'])
    with col4:
        st.metric("Missing Values", results['summary']['missing_values'])
    
    # Errors
    if results['errors']:
        st.error("❌ Validation Errors:")
        for error in results['errors']:
            st.error(f"• {error}")
    
    # Warnings
    if results['warnings']:
        st.warning("⚠️ Data Quality Warnings:")
        for warning in results['warnings']:
            st.warning(f"• {warning}")
    
    if results['is_valid'] and not results['warnings']:
        st.success("✅ All validations passed!")


def process_data(df: pd.DataFrame) -> None:
    """Process and save validated data to database."""
    insert_data("spend_data", df)
    st.success("Spend data saved!")
