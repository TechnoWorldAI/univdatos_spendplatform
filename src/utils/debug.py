"""
Debug utilities for the Spend Platform application.
Provides centralized logging and error handling functionality.
"""

import logging
import traceback
import streamlit as st
from datetime import datetime
from typing import Any, Optional, Dict
import functools
import inspect


class DebugLogger:
    """Centralized debug logger for the application."""
    
    def __init__(self):
        self.logger = logging.getLogger('spend_platform')
        if not self.logger.handlers:
            self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
    
    def debug(self, message: str, extra_data: Optional[Dict] = None):
        """Log debug message with optional extra data."""
        try:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_info = f"{frame.f_back.f_code.co_filename}:{frame.f_back.f_lineno}"
            else:
                caller_info = "unknown"
        except Exception:
            caller_info = "unknown"
        
        log_msg = f"[{caller_info}] {message}"
        if extra_data:
            log_msg += f" | Data: {extra_data}"
        
        self.logger.debug(log_msg)
        
        # Also show in Streamlit if in debug mode - guard for non-streamlit contexts
        try:
            debug_mode = False
            if hasattr(st, 'session_state'):
                debug_mode = bool(st.session_state.get('debug_mode', False))
        except Exception:
            debug_mode = False

        if debug_mode:
            try:
                with st.sidebar:
                    st.text(f"🐛 DEBUG: {message}")
                    if extra_data:
                        st.json(extra_data)
            except Exception:
                # If Streamlit UI isn't available, skip UI debug display
                pass
    
    def info(self, message: str, extra_data: Optional[Dict] = None):
        """Log info message."""
        self.logger.info(message)
        if extra_data:
            self.logger.info(f"Extra data: {extra_data}")
    
    def warning(self, message: str, extra_data: Optional[Dict] = None):
        """Log warning message."""
        self.logger.warning(message)
        if extra_data:
            self.logger.warning(f"Extra data: {extra_data}")
    
    def error(self, message: str, exception: Optional[Exception] = None, extra_data: Optional[Dict] = None):
        """Log error message with optional exception details."""
        self.logger.error(message)
        if exception:
            self.logger.error(f"Exception: {str(exception)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        if extra_data:
            self.logger.error(f"Extra data: {extra_data}")

    def exception(self, message: str, exception: Optional[Exception] = None, extra_data: Optional[Dict] = None):
        """Log an exception with full traceback. Use inside except blocks or when an Exception is available."""
        # Use logger.exception to capture stack trace when called inside an exception handler
        try:
            if exception is not None:
                # Log a clear message then include traceback
                self.logger.error(message)
                self.logger.error(f"Exception: {str(exception)}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
            else:
                # If no exception object provided, still log message at exception level
                self.logger.exception(message)

            if extra_data:
                self.logger.error(f"Extra data: {extra_data}")
        except Exception:
            # Fallback to basic error logging if something goes wrong while logging
            self.logger.error(f"Failed to log exception for message: {message}")


# Global debug logger instance
debug_logger = DebugLogger()


def show_error_block(title: str, error: Exception, show_details: bool = True):
    """
    Display a user-friendly error block in Streamlit.
    
    Args:
        title: Error title to display
        error: The exception that occurred
        show_details: Whether to show technical details
    """
    # Guard Streamlit UI calls so this function can be used from scripts
    try:
        st.error(f"❌ **{title}**")

        # Always show the main error message
        st.error(f"**Error:** {str(error)}")

        if show_details:
            try:
                with st.expander("🔧 Technical Details"):
                    st.code(f"Exception Type: {type(error).__name__}")
                    st.code(f"Error Message: {str(error)}")

                    # Show traceback if available
                    tb = traceback.format_exc()
                    if tb != "NoneType: None\n":
                        st.code(tb)

                    # Show timestamp
                    st.text(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception:
                # If expanding UI isn't available, skip UI details
                pass
    except Exception:
        # Not running inside Streamlit — fall back to logging
        debug_logger.error(f"{title}: {str(error)}", exception=error)

    # Log the error with full traceback
    debug_logger.exception(title, error)


def debug_function_call(func_name: str, args: tuple = (), kwargs: Optional[dict] = None):
    """Debug function entry point."""
    kwargs = kwargs or {}
    debug_logger.debug(
        f"Function '{func_name}' called",
        {"args": str(args), "kwargs": kwargs}
    )


def debug_function_result(func_name: str, result: Any):
    """Debug function result."""
    debug_logger.debug(
        f"Function '{func_name}' returned",
        {"result_type": type(result).__name__, "result": str(result)[:200]}
    )


def debug_decorator(func):
    """Decorator to automatically debug function calls."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        debug_function_call(func_name, args, kwargs)
        
        try:
            result = func(*args, **kwargs)
            debug_function_result(func_name, result)
            return result
        except Exception as e:
            debug_logger.exception(f"Exception in {func_name}", e)
            raise
    
    return wrapper


def safe_execute(func, *args, error_title: str = "Operation Failed", show_ui_error: bool = True, **kwargs):
    """
    Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        error_title: Title to show in UI error
        show_ui_error: Whether to show error in UI
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (success: bool, result: Any, error: Optional[Exception])
    """
    try:
        debug_function_call(func.__name__, args, kwargs)
        result = func(*args, **kwargs)
        debug_function_result(func.__name__, result)
        return True, result, None
        
    except Exception as e:
        debug_logger.exception(f"Error in {func.__name__}", e)

        if show_ui_error:
            show_error_block(error_title, e)

        return False, None, e


def enable_debug_mode():
    """Enable debug mode in session state."""
    try:
        st.session_state['debug_mode'] = True
        st.success("🐛 Debug mode enabled")
    except Exception:
        debug_logger.info("Debug mode enabled (non-UI context)")


def disable_debug_mode():
    """Disable debug mode in session state."""
    try:
        st.session_state['debug_mode'] = False
        st.success("Debug mode disabled")
    except Exception:
        debug_logger.info("Debug mode disabled (non-UI context)")


def debug_session_state():
    """Show current session state for debugging."""
    try:
        debug_mode = bool(st.session_state.get('debug_mode', False))
    except Exception:
        debug_mode = False

    if debug_mode:
        try:
            with st.sidebar:
                st.subheader("🔍 Debug Info")

                # Show session state
                with st.expander("Session State"):
                    st.json(dict(st.session_state))

                # Debug mode toggle
                if st.button("Disable Debug Mode"):
                    disable_debug_mode()
        except Exception:
            # Non-UI context — nothing to show
            pass
    else:
        # Show option to enable debug mode when UI is available
        try:
            if st.sidebar.button("🐛 Enable Debug Mode"):
                enable_debug_mode()
        except Exception:
            # Non-UI context — no interactive control
            pass


def log_page_load(page_name: str):
    """Log when a page is loaded."""
    debug_logger.info(f"Page loaded: {page_name}")
    
    if st.session_state.get('debug_mode', False):
        st.sidebar.success(f"📄 Loaded: {page_name}")


# Alias for backward compatibility
Debug = DebugLogger
