"""Login page module for user authentication."""
import streamlit as st
from src.services.auth_service import AuthService
from src.exceptions.base import AuthenticationError
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the login page."""
    try:
        # Inject custom CSS for login form styling
        st.markdown(
            '''<style>
            /* Hide Streamlit sidebar on login page */
            section[data-testid="stSidebar"] {
                width: 0 !important;
                min-width: 0 !important;
                max-width: 0 !important;
                overflow: hidden !important;
                display: none !important;
            }
            body {
                background: radial-gradient(circle, #b3c6e7 0%, #7fa6d6 100%);
            }
            .stApp {
                background: radial-gradient(circle, #b3c6e7 0%, #7fa6d6 100%) !important;
            }
            div[data-testid="stForm"] {
                background: rgba(40, 70, 120, 0.85);
                border-radius: 16px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                padding: 32px 24px 24px 24px;
                margin: 60px auto 0 auto;
                max-width: 350px;
                color: #fff;
            }
            label, .stTextInput label, .stPasswordInput label {
                color: #fff !important;
                font-size: 1.1em;
            }
            input[type="text"], input[type="password"] {
                border-radius: 8px;
                border: none;
                background: #eaf0fa;
                padding: 10px;
                margin-bottom: 0px;
                font-size: 1em;
            }
            button[kind="primary"], button[data-testid="baseButton-primary"], .stButton > button {
                background: #2176ae !important;
                color: #fff !important;
                border-radius: 8px;
                border: none;
                box-shadow: 0 2px 8px rgba(31, 38, 135, 0.2);
                font-size: 1.1em;
                padding: 10px 0;
                margin-top: 10px;
                display: block;
                margin-left: auto;
                margin-right: auto;
                transition: none !important;
            }
            input[type="checkbox"] {
                accent-color: #5ec6e7;
                width: 18px;
                height: 18px;
                border-radius: 4px;
                margin-right: 8px;
            }
            </style>''', unsafe_allow_html=True)
        debug_logger.info("Rendering login page")
        
        # Clear sidebar completely
        st.sidebar.empty()
        with st.sidebar:
            st.markdown("### 🔒 Please Log In")
            # st.markdown("Sidebar cleared for unauthenticated users")  
        
        
        # st.markdown("**This is the NEW refactored application - PORT 8502**")
        # st.success("✅ SUCCESS: Sidebar navigation is hidden before authentication!")
        
        with st.form("login_form"):
            st.title("Member Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                debug_logger.debug("Login form submitted", {"username": username})
                
                try:
                    auth_service = AuthService()
                    debug_logger.debug("AuthService initialized")
                    
                    user_info = auth_service.authenticate(username, password)
                    debug_logger.info("Authentication successful", {"username": username, "role": user_info.get('role')})
                    
                    # Set session state
                    st.session_state.authenticated = True
                    # Ensure the selected page defaults to dashboard for the new user
                    st.session_state.selected_page = "dashboard"
                    st.session_state.user = user_info
                    debug_logger.debug("Session state updated for authenticated user")
                    
                    st.success("✅ Login successful!")
                    st.rerun()
                    
                except AuthenticationError as e:
                    debug_logger.warning("Authentication failed", extra_data={"username": username, "error": str(e)})
                    st.error("❌ Invalid username or password")
                except Exception as e:
                    debug_logger.exception("Unexpected error during authentication", e, {"username": username})
                    show_error_block("Login Error", e)
        
    # Demo credentials help (hidden for now)
    # with st.expander("ℹ️ Demo Credentials"):
    #         st.markdown("""
    #         The demo password policy has been updated for this environment.
    #
    #         Use the pattern: `username1234` (for example, `admin` -> `admin1234`).
    #
    #         - **Admin**: `admin` / `admin1234`
    #         - **Spend Manager**: `manager` / `manager1234`
    #         - **Data Analyst**: `analyst` / `analyst1234`
    #         """)
            
    except Exception as e:
        debug_logger.exception("Error rendering login page", e)
        show_error_block("Login Page Error", e)
