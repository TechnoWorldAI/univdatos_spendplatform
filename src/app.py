
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
from src.utils.debug import debug_logger

logger = debug_logger
# Inject custom CSS for sidebar navigation styling
st.markdown("""
<style>
body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    background-color: #FAFAFA !important;
}
/* Page border for main app container */
[data-testid="stAppViewContainer"] {
    border: 1px solid black !important;
    border-radius: 6px !important;
    box-sizing: border-box !important;
    margin: 6px !important;
}
/* 1) Sidebar background color */
[data-testid="stSidebar"] {
    background-color: #1D2951 !important;
}
/* 2) Style any buttons placed in the sidebar */
 [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        padding: 0.6rem 1rem;
        color: #fff;
        font-weight: 600;
        text-align: left;
        transition: transform 0.02s ease-in-out;
        background: none !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        border-radius: 0 !important;
        display: flex;
        justify-content: flex-start;
        align-items: center;
}
[data-testid="stSidebar"] .stButton > button:hover {
        box-shadow: none !important;
        background: rgba(0,0,0,0.04) !important;
}
[data-testid="stSidebar"] .stButton > button:active {
        transform: translateY(1px);
        box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)    

# Render logo and title in a flexbox row at the top of the main page
def get_logo_img_tag():
    logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'context', 'UDS_Logo.PNG')
    if os.path.exists(logo_path):
        import base64
        with open(logo_path, "rb") as image_file:
            encoded_logo = base64.b64encode(image_file.read()).decode()
        return f"<img src='data:image/png;base64,{encoded_logo}' style='height:20px; margin-right:20px; vertical-align:middle;'>"
    return ""
def get_contact_img_tag():
    contact_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'context', 'Contact.png')
    if os.path.exists(contact_path):
        import base64
        with open(contact_path, "rb") as image_file:
            encoded_contact = base64.b64encode(image_file.read()).decode()
        return f"<img src='data:image/png;base64,{encoded_contact}' style='height:50px; margin-left:20px; vertical-align:middle;'>"
    return ""
logo_img_tag = get_logo_img_tag()
contact_img_tag = get_contact_img_tag()
st.markdown(f"""
<div style='position: relative; width: 100%; height: 0px; margin-bottom:40px;'>
    <div style='position: absolute; left: 0; top: 50%; transform: translateY(-50%);'>
        {logo_img_tag}
    </div>
    <div style='width:100%; display:flex; justify-content:center; align-items:center; height:100%;'>
        <span style='font-size:1.8em; color:#002366; font-weight:600; text-align:center;'>Spend Analysis Platform</span>
    </div>
    <div style='position: absolute; right: 0; top: 50%; transform: translateY(-50%);'>
        {contact_img_tag}
    </div>
</div>
""", unsafe_allow_html=True)
st.markdown("<div style='margin-top:0px;'></div>", unsafe_allow_html=True)
st.markdown("<hr style='border:0.5px solid #B0B0B0; margin:0px;'>", unsafe_allow_html=True)
        # Load data


def init_app():
    """Initialize application configuration"""
    logger.info("Starting main application")
    logger.info("Initializing application")
    
    # Page configuration - IMPORTANT: Set this before any other Streamlit commands
    st.set_page_config(
        page_title="Spend Platform",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"  # Sidebar always open
    )
    
    # CRITICAL: Hide automatic page discovery by clearing any auto-generated navigation
    # This prevents Streamlit from automatically showing pages from src/pages/ directory
    st.session_state._pages = {}
    
    logger.debug("Page config set successfully")
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_exists' not in st.session_state:
        st.session_state.user_exists = False
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    logger.debug(f"Session state initialized | Data: {{'authenticated': {st.session_state.authenticated}, 'user_exists': {st.session_state.user_exists}, 'debug_mode': {st.session_state.debug_mode}}}")

def sidebar_navigation():
    """Render sidebar navigation"""
    from src.services.role_service import RoleService
    from src.services.auth_service import AuthService
    
    logger.debug("Rendering sidebar navigation")    
    
    # Display logo and user info
    if st.session_state.authenticated:
        with st.sidebar:
            #render_sidebar_logo()
            user_info = st.session_state.get('user', {})
            username = user_info.get('username', 'User')
            user_role = user_info.get('role', 'Unknown')
            st.markdown(f"<span style='color:#fff; font-weight: bold; display:block;'>👤 {username}</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='color:#fff; font-size:0.85em; padding-left:22px; margin-bottom:-8px;'> {user_role}</span>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:6px 0 12px 0; border-top:1px solid #fff; opacity:0.25;'>", unsafe_allow_html=True)
            logger.debug(f"User info displayed | Data: {{'username': '{username}', 'role': '{user_role}'}}")
            
            # Initialize services
            role_service = RoleService()
            auth_service = AuthService()
            
            # Check permissions - use actual user role, not fallback
            permissions = role_service.get_role_permissions(user_role)
            
            logger.debug(f"Checking user permissions | Data: {{'role': '{user_role}', 'permissions': {permissions}}}")
            
            # Set default page if none selected
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = "dashboard"
            
            # Navigation buttons based on permissions
            # Dashboard - available to all authenticated users
            if st.button("🏠  Dashboard", key="nav_dashboard", width='stretch'):
                st.session_state.selected_page = "dashboard"
                st.rerun()

            # Data Upload
            if permissions.get("can_upload_data", False):
                if st.button("📤  Upload Spend Data", key="nav_upload",width='stretch'):
                    st.session_state.selected_page = "upload"
                    st.rerun()

            # Categorization    
            if st.button("📂 Categorize Items", key="nav_categorize",width='stretch'):
                    st.session_state.selected_page = "categorize"
                    st.rerun()

            # User Management
            if permissions.get("can_manage_users", False):
                if st.button("👥 User Management", key="nav_users", width='stretch'):
                    st.session_state.selected_page = "users"
                    st.rerun()
            
            # Rules Management
            if permissions.get("can_manage_rules", False):
                if st.button("⚙️ Rules", key="nav_rules", width='stretch'):
                    st.session_state.selected_page = "rules"
                    st.rerun()
            
            # Error Resolution
            if permissions.get("can_resolve_errors", False):
                if st.button("🔧 Error Resolution", key="nav_errors", width='stretch'):
                    st.session_state.selected_page = "errors"
                    st.rerun()
            
            # Master Data Management (split into Vendors and Categories)
            #if permissions.get("can_manage_master_data", False):
            #    if st.button("�  Vendors", key="nav_vendors", width='stretch'):
            #        st.session_state.selected_page = "vendors"
            #        st.rerun()
            

            # Categorization upload/import
            if permissions.get("can_manage_master_data", False):
                if st.button("📥 Categorization Upload", key="nav_categorization_upload"):
                    st.session_state.selected_page = "categorization_upload"
                    st.rerun()
            
            # Categories
            if st.button("📂 Categories", key="nav_categories",width='stretch'):
                st.session_state.selected_page = "categories"
                st.rerun()

            # Logout button
            st.markdown("---")
            if st.button("🚪 Logout", key="logout",width='stretch'):
                auth_service.logout()
                st.rerun()
            
            logger.debug(f"Navigation completed | Data: {{'selected_page': '{st.session_state.selected_page}'}}")
            
            return st.session_state.selected_page
    
    return None

def clear_sidebar():
    """Clear all sidebar content"""
    logger.debug("Clearing sidebar for unauthenticated user")
    # This ensures the sidebar is completely empty
    with st.sidebar:
        st.empty()  # This should clear all sidebar content
    logger.debug("Sidebar cleared successfully")

def main():
    """Main application function"""
    init_app()
    
    # Initialize database - we don't need to import DatabaseManager
    logger.info("Initializing database")
    logger.info("Database initialization completed")
    
    # Check authentication status
    if not st.session_state.authenticated:
        logger.debug("User not authenticated, showing login page")
        # Important: Clear sidebar completely for unauthenticated users
        clear_sidebar()
        selected_page = "login"
        page_title = "Login"
    else:
        logger.debug("User authenticated, proceeding to navigation")
        selected_page = sidebar_navigation()
        page_title = selected_page.title() if selected_page else "Dashboard"
    
    logger.info(f"Page loaded: {page_title}")
    
    # Render selected page
    if selected_page == "login":
        from src.pages_modules.login import render_page
        logger.info("Rendering login page")
        render_page()
    
    elif selected_page == "dashboard":
        from src.pages_modules.dashboard import render_page
        logger.info("Rendering dashboard page")
        render_page()
    
    elif selected_page == "upload":
        from src.pages_modules.upload import render_page
        logger.info("Rendering upload page")
        render_page()

    elif selected_page == "categorize":
        from src.pages_modules.categorize import render_page
        logger.info("Rendering categorize page")
        render_page()

    elif selected_page == "vendors":
        from src.pages_modules.vendor_management import render_page
        logger.info("Rendering vendor management page")
        render_page()

    elif selected_page == "categories":
        from src.pages_modules.category_management import render_page
        logger.info("Rendering category management page")
        render_page()
    
    elif selected_page == "users":
        from src.pages_modules.user_management import render_page
        logger.info("Rendering user management page")
        render_page()
    
    elif selected_page == "rules":
        from src.pages_modules.rules import render_page
        logger.info("Rendering rules management page")
        render_page()
    
    elif selected_page == "errors":
        from src.pages_modules.error_management import render_page
        logger.info("Rendering error resolution page")
        render_page()
    
    elif selected_page == "categorization_upload":
        from src.pages_modules.categorization_upload import render_page
        logger.info("Rendering categorization upload page")
        render_page()

if __name__ == "__main__":
    main()
