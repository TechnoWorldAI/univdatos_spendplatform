"""Error management page module."""
import streamlit as st
import pandas as pd
import logging
import traceback
from datetime import datetime, timedelta
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute, enable_debug_mode, disable_debug_mode
from src.utils.display import normalize_df_for_display


def render_page() -> None:
    """Render the error management and debug console page."""
    try:
        debug_logger.info("Rendering error management page")
        st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> 🔧 Error Management</div>", unsafe_allow_html=True)
        
        # Create tabs for different error management functions
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Data Errors", "🐛 Debug Console", "📊 System Health", "⚙️ Settings"])
        
        with tab1:
            render_data_errors()
            
        with tab2:
            render_debug_console()
            
        with tab3:
            render_system_health()
            
        with tab4:
            render_debug_settings()
            
    except Exception as e:
        debug_logger.error("Error rendering error management page", e)
        show_error_block("Error Management Page Error", e)


def render_data_errors():
    """Render data errors section."""
    
    st.markdown(f"<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> Data Processing Errors </div>", unsafe_allow_html=True)

    # Load error data
    success, errors_df, error = safe_execute(
        load_errors,
        error_title="Failed to Load Errors",
        show_ui_error=True
    )
    
    if not success or errors_df is None:
        return
    
    if errors_df.empty:
        st.info("No data processing errors found. Great job!")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Errors", len(errors_df))
    with col2:
        open_errors = errors_df[errors_df['status'] == 'Open']
        st.metric("Open Errors", len(open_errors))
    with col3:
        resolved_errors = errors_df[errors_df['status'] == 'Resolved']
        st.metric("Resolved Errors", len(resolved_errors))
    
    # Filter options
    with st.expander("🔍 Filter Options"):
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.multiselect(
                "Status", 
                options=errors_df['status'].unique(),
                default=errors_df['status'].unique()
            )
        with col2:
            error_type_filter = st.multiselect(
                "Error Type",
                options=errors_df['error_type'].unique(),
                default=errors_df['error_type'].unique()
            )
    
    # Apply filters
    filtered_df = errors_df[
        (errors_df['status'].isin(status_filter)) &
        (errors_df['error_type'].isin(error_type_filter))
    ]
    
    # Display errors
    st.markdown(f"<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> Errors ({len(filtered_df)}) </div>", unsafe_allow_html=True)

    if not filtered_df.empty:
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Mark All as Resolved"):
                resolve_all_errors(filtered_df['error_id'].tolist())
                st.success("All errors marked as resolved!")
                st.rerun()
        
        # Display error table
        st.dataframe(
            filtered_df,
            width='stretch',
            column_config={
                "error_id": "ID",
                "transaction_id": "Transaction ID",
                "error_type": "Type",
                "description": "Description",
                "status": "Status",
                "created_at": "Created",
                "resolved_at": "Resolved"
            }
        )


def load_errors() -> pd.DataFrame:
    """Load error data from database."""
    with get_db_connection() as conn:
        query = """
            SELECT 
                e.*,
                st.supplier_name,
                st.item_invoice_value
            FROM error_logs e
            LEFT JOIN spend_transactions st ON e.transaction_id = st.transaction_id
            ORDER BY e.created_at DESC
        """
        return pd.read_sql_query(query, conn)


def resolve_all_errors(error_ids: list) -> None:
    """Resolve multiple errors."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        for error_id in error_ids:
            cursor.execute("""
                UPDATE error_logs 
                SET status = 'Resolved', 
                    resolved_at = datetime('now'),
                    resolved_by = ?
                WHERE error_id = ?
            """, ('system', error_id))
        
        conn.commit()


def render_debug_console():
    """Render debug console section."""
    st.markdown(f"<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> 🐛 Debug Console </div>", unsafe_allow_html=True)
    # Debug mode toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.get('debug_mode', False):
            if st.button("🔴 Disable Debug Mode", key="disable_debug"):
                disable_debug_mode()
                st.rerun()
        else:
            if st.button("🟢 Enable Debug Mode", key="enable_debug"):
                enable_debug_mode()
                st.rerun()
    
    with col2:
        if st.button("🔄 Clear Logs", key="clear_logs"):
            # Clear the in-memory logs (this is a placeholder)
            st.success("Logs cleared!")
    
    # Show current session state
    with st.expander("📊 Current Session State"):
        st.json(dict(st.session_state))
    
    # Show recent log entries (if available)
    st.subheader("Recent Log Entries")
    
    # Mock log entries for demonstration
    log_entries = [
        {"timestamp": "2025-08-30 12:22:20", "level": "INFO", "message": "Application initialized successfully"},
        {"timestamp": "2025-08-30 12:22:15", "level": "DEBUG", "message": "Database connection established"},
        {"timestamp": "2025-08-30 12:22:10", "level": "WARNING", "message": "Slow query detected: load_vendors took 2.3s"},
        {"timestamp": "2025-08-30 12:22:05", "level": "ERROR", "message": "Failed to validate data: Missing required column 'amount'"},
    ]
    
    for entry in log_entries:
        level_color = {
            "DEBUG": "🔵",
            "INFO": "🟢", 
            "WARNING": "🟡",
            "ERROR": "🔴"
        }.get(entry["level"], "⚪")
        
        st.text(f"{level_color} {entry['timestamp']} | {entry['level']} | {entry['message']}")


def render_system_health():
    """Render system health monitoring section."""
    st.markdown(f"<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> 📊 System Health </div>", unsafe_allow_html=True)
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Database Status", "🟢 Connected", delta="Normal")
    
    with col2:
        st.metric("Active Users", "1", delta="0")
        
    with col3:
        st.metric("Memory Usage", "45%", delta="-5%")
        
    with col4:
        st.metric("Response Time", "120ms", delta="10ms")
    
    # Database statistics
    st.markdown(f"<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> 📈 Database Statistics </div>", unsafe_allow_html=True)
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get table statistics with single optimized query
            query = """
                SELECT 
                    'spend_transactions' as table_name, COUNT(*) as count FROM spend_transactions
                UNION ALL
                SELECT 'vendors' as table_name, COUNT(*) as count FROM vendors
                UNION ALL  
                SELECT 'categories' as table_name, COUNT(*) as count FROM categories
                UNION ALL
                SELECT 'error_logs' as table_name, COUNT(*) as count FROM error_logs
            """
            
            try:
                cursor.execute(query)
                results = cursor.fetchall()
                
                stats_data = []
                for row in results:
                    table_name, count = row
                    stats_data.append({"Table": table_name, "Row Count": count})
                    
            except Exception as e:
                # Fallback to individual queries if UNION fails
                debug_logger.exception("Failed to execute optimized count query, falling back to individual queries", e)
                tables = ['spend_transactions', 'vendors', 'categories', 'error_logs']
                stats_data = []
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        stats_data.append({"Table": table, "Row Count": count})
                    except Exception as e:
                        debug_logger.exception(f"Failed to get count for table {table}", e)
                        stats_data.append({"Table": table, "Row Count": "N/A"})
            
            stats_df = pd.DataFrame(stats_data)
            try:
                st.dataframe(normalize_df_for_display(stats_df), width='stretch')
            except Exception:
                st.dataframe(stats_df, width='stretch')
            
    except Exception as e:
        st.error(f"Failed to load database statistics: {str(e)}")
    
    # Recent activity
    st.markdown(f"<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> 🕒 Recent Activity </div>", unsafe_allow_html=True)
    activity_data = [
        {"Time": "12:22:20", "Action": "User logged in", "User": "admin"},
        {"Time": "12:22:15", "Action": "Data uploaded", "User": "admin"},
        {"Time": "12:22:10", "Action": "Vendor added", "User": "admin"},
    ]
    
    activity_df = pd.DataFrame(activity_data)
    try:
        st.dataframe(normalize_df_for_display(activity_df), width='stretch')
    except Exception:
        st.dataframe(activity_df, width='stretch')


def render_debug_settings():
    """Render debug settings and configuration."""
    st.markdown(f"<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> ⚙️ Debug Settings </div>", unsafe_allow_html=True)
    # Debug level settings
    st.write("**Logging Configuration**")
    
    current_level = st.selectbox(
        "Log Level",
        options=["DEBUG", "INFO", "WARNING", "ERROR"],
        index=0,
        help="Set the minimum log level to capture"
    )
    
    # Debug features
    st.write("**Debug Features**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        show_sql_queries = st.checkbox("Show SQL Queries", value=False)
        show_function_calls = st.checkbox("Show Function Calls", value=True)
        show_performance_metrics = st.checkbox("Show Performance Metrics", value=False)
    
    with col2:
        enable_error_details = st.checkbox("Show Error Details", value=True)
        enable_stack_traces = st.checkbox("Show Stack Traces", value=True)
        enable_session_tracking = st.checkbox("Track Session Changes", value=False)
    
    if st.button("💾 Save Settings"):
        # Save debug settings to session state
        st.session_state.update({
            'debug_log_level': current_level,
            'debug_show_sql': show_sql_queries,
            'debug_show_functions': show_function_calls,
            'debug_show_performance': show_performance_metrics,
            'debug_show_errors': enable_error_details,
            'debug_show_traces': enable_stack_traces,
            'debug_track_session': enable_session_tracking,
        })
        st.success("Debug settings saved!")
    
    # Test debug features
    st.write("**Test Debug Features**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧪 Test Info Log"):
            debug_logger.info("This is a test info message")
            st.success("Info log sent!")
    
    with col2:
        if st.button("⚠️ Test Warning Log"):
            debug_logger.warning("This is a test warning message")
            st.warning("Warning log sent!")
    
    with col3:
        if st.button("❌ Test Error Log"):
            try:
                # Intentionally cause an error for testing
                raise ValueError("This is a test error for debugging purposes")
            except Exception as e:
                debug_logger.exception("Test error generated", e)
                show_error_block("Test Error", e, show_details=True)
