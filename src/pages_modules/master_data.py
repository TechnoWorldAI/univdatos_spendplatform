"""Master data management page module."""
import streamlit as st
import pandas as pd
from typing import Optional
from src.utils.db_simple import get_db_connection
from src.utils.display import normalize_df_for_display
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the master data management page."""
    try:
        debug_logger.info("Rendering master data management page")
        st.title("🗂️ Master Data Management")
        st.markdown("### Central data management for vendors and categories")
        
        tab1, tab2 = st.tabs(["📦 Vendors", "📂 Categories"])
        
        with tab1:
            debug_logger.debug("Rendering vendor management tab")
            render_vendor_management()
        
        with tab2:
            debug_logger.debug("Rendering category management tab")
            render_category_management()
            
    except Exception as e:
        debug_logger.exception("Error rendering master data page", e)
        show_error_block("Master Data Page Error", e)


def render_vendor_management() -> None:
    """Render vendor management interface."""
    try:
        debug_logger.debug("Starting vendor management rendering")
        st.subheader("Vendor Management")
        
        # Add new vendor form
        debug_logger.debug("Creating vendor form section")
        st.subheader("➕ Add New Vendor")
        debug_logger.debug("Created vendor form subheader")
        
        with st.form("add_vendor"):
            debug_logger.debug("Inside vendor form context")
            vendor_name = st.text_input("Vendor Name")
            vendor_code = st.text_input("Vendor Code")
            contact_email = st.text_input("Contact Email")
            debug_logger.debug("Vendor form inputs created", {
                "form_name": "add_vendor",
                "has_vendor_name": bool(vendor_name),
                "has_vendor_code": bool(vendor_code),
                "has_contact_email": bool(contact_email)
            })
            
            submit_button = st.form_submit_button("Add Vendor")
            debug_logger.debug("Form submit button created", {"button_clicked": submit_button})
            
            if submit_button:
                debug_logger.info("🎯 FORM SUBMISSION DETECTED! Add vendor button clicked", {
                    "vendor_name": vendor_name,
                    "vendor_code": vendor_code,
                    "has_email": bool(contact_email),
                    "vendor_name_length": len(vendor_name) if vendor_name else 0,
                    "submit_button_state": submit_button
                })
                
                if vendor_name and vendor_name.strip():
                    debug_logger.info("✅ Form validation passed - proceeding to add vendor", {
                        "vendor_name": vendor_name.strip(),
                        "vendor_code": vendor_code.strip() if vendor_code else None,
                        "contact_email": contact_email.strip() if contact_email else None
                    })
                    
                    success, result, error = safe_execute(
                        lambda **kwargs: add_vendor(
                            vendor_name.strip(),
                            vendor_code.strip() if vendor_code else None,
                            contact_email.strip() if contact_email else None
                        )
                    )
                    
                    if success and result:
                        st.success(f"✅ Vendor '{vendor_name}' added successfully!")
                        debug_logger.info("✅ Vendor added successfully - staying on current page", {
                            "vendor_name": vendor_name,
                            "new_vendor_id": result
                        })
                    else:
                        error_msg = str(error) if error else "Failed to add vendor"
                        st.error(f"❌ Error adding vendor: {error_msg}")
                        debug_logger.error("Failed to add vendor from form", None, {"error": error_msg, "vendor_name": vendor_name})
                else:
                    st.error("❌ Please enter a vendor name")
                    debug_logger.warning("Form submission attempted without vendor name", {
                        "vendor_name_provided": bool(vendor_name),
                        "vendor_name_stripped": bool(vendor_name.strip()) if vendor_name else False
                    })
            else:
                debug_logger.debug("Form submit button not clicked (normal state)")
        
        debug_logger.debug("Vendor form rendering completed")
        
        # Display existing vendors
        debug_logger.debug("Loading existing vendors")
        success, vendors_df, error = safe_execute(
            load_vendors,
            error_title="Failed to Load Vendors",
            show_ui_error=True
        )
        
        if success and vendors_df is not None and not vendors_df.empty:
            debug_logger.debug("Displaying vendors", {"count": len(vendors_df)})
            st.subheader(f"Existing Vendors ({len(vendors_df)})")
            try:
                st.dataframe(normalize_df_for_display(vendors_df), width='stretch')
            except Exception:
                st.dataframe(vendors_df, width='stretch')
        elif success:
            debug_logger.debug("No vendors found")
            st.info("No vendors found. Add some vendors to get started.")
            
    except Exception as e:
        debug_logger.exception("Error in vendor management rendering", e)
        show_error_block("Vendor Management Error", e)


def render_category_management() -> None:
    """Render category management interface."""
    try:
        debug_logger.debug("Starting category management rendering")
        st.subheader("Category Management")
        
        # Add new category form
        with st.expander("➕ Add New Category"):
            with st.form("add_category"):
                category_name = st.text_input("Category Name")
                description = st.text_area("Description", height=80)

                # Load parent categories safely
                success, parent_categories, error = safe_execute(
                    load_categories,
                    error_title="Failed to Load Categories",
                    show_ui_error=False  # Don't show error for initial load
                )

                # Build options as (id, name) tuples so we can reliably map selection to ID
                parent_option_tuples = [(None, "None")]
                if success and parent_categories is not None and not parent_categories.empty:
                    parent_option_tuples += list(zip(parent_categories['category_id'].tolist(), parent_categories['category_name'].tolist()))

                selected_parent = st.selectbox("Parent Category", parent_option_tuples, format_func=lambda x: x[1])
                # selected_parent is a tuple (id, name)
                parent_id = None
                try:
                    parent_id = selected_parent[0] if selected_parent is not None else None
                except Exception:
                    parent_id = None

                debug_logger.debug("Category form initialized", {"parent_options_count": len(parent_option_tuples), "selected_parent_id": parent_id})

                if st.form_submit_button("Add Category"):
                    debug_logger.debug("Add category form submitted", {
                        "category_name": category_name,
                        "parent_id": parent_id,
                        "description_provided": bool(description)
                    })

                    if category_name:
                        success, result, error = safe_execute(
                            add_category,
                            category_name, parent_id, description,
                            error_title="Failed to Add Category",
                            show_ui_error=True
                        )

                        if success:
                            debug_logger.info("Category added successfully", {"category_name": category_name, "parent_id": parent_id})
                            st.success("✅ Category added successfully!")
                            # Use experimental_rerun if available for better compatibility
                            rerun_fn = getattr(st, 'experimental_rerun', None)
                            if callable(rerun_fn):
                                rerun_fn()
                            else:
                                try:
                                    st.rerun()
                                except Exception:
                                    pass
                    else:
                        debug_logger.warning("Add category attempt with empty name")
                        st.error("❌ Category name is required")
        
        # Display existing categories
        debug_logger.debug("Loading existing categories")
        success, categories_df, error = safe_execute(
            load_categories,
            error_title="Failed to Load Categories",
            show_ui_error=True
        )
        
        if success and categories_df is not None and not categories_df.empty:
            debug_logger.debug("Displaying categories", {"count": len(categories_df)})
            st.subheader(f"Existing Categories ({len(categories_df)})")
            try:
                st.dataframe(normalize_df_for_display(categories_df), width='stretch')
            except Exception:
                st.dataframe(categories_df, width='stretch')
        elif success:
            debug_logger.debug("No categories found")
            st.info("No categories found. Add some categories to get started.")
            
    except Exception as e:
        debug_logger.exception("Error in category management rendering", e)
        show_error_block("Category Management Error", e)


def load_vendors() -> pd.DataFrame:
    """Load vendors from database."""
    debug_logger.debug("Loading vendors from database")
    
    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query("""
                SELECT vendor_id, vendor_name, supplier_no, contact_info, created_at, updated_at
                FROM vendors
                ORDER BY created_at DESC
            """, conn)
            
            debug_logger.debug("Vendors loaded successfully", {
                "count": len(df),
                "latest_vendor": df.iloc[0]['vendor_name'] if not df.empty else None
            })
            
            return df
    except Exception as e:
        debug_logger.exception("Error loading vendors", e)
        raise Exception(f"Database error loading vendors: {str(e)}")


def load_categories() -> pd.DataFrame:
    """Load category data from database.
    
    Returns:
        DataFrame containing category data
        
    Raises:
        Exception: If database operation fails
    """
    debug_logger.debug("Loading categories from database")
    try:
        with get_db_connection() as conn:
            query = "SELECT * FROM categories ORDER BY category_name"
            df = pd.read_sql_query(query, conn)
            debug_logger.debug("Categories loaded successfully", {"count": len(df)})
            return df
    except Exception as e:
        debug_logger.exception("Error loading categories", e)
        raise Exception(f"Database error loading categories: {str(e)}")


def add_vendor(name: str, code: Optional[str] = None, email: Optional[str] = None) -> int:
    """Add a new vendor to the database.
    
    Args:
        name: Vendor name (required)
        code: Optional vendor code (stored as supplier_no)
        email: Optional contact email (stored as contact_info)
        
    Returns:
        int: The vendor_id of the newly created vendor
        
    Raises:
        Exception: If database operation fails
    """
    debug_logger.debug("Adding new vendor to database", {
        "name": name,
        "code": code,
        "has_email": bool(email),
        "normalized_name": name.lower().strip()
    })
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check if vendor already exists
            cursor.execute("SELECT vendor_id FROM vendors WHERE normalized_name = ?", (name.lower().strip(),))
            existing = cursor.fetchone()
            
            if existing:
                debug_logger.warning("Vendor already exists", {"name": name, "existing_id": existing[0]})
                raise Exception(f"Vendor '{name}' already exists")
            
            # Insert new vendor
            debug_logger.debug("Inserting vendor into database", {"name": name})
            cursor.execute("""
                INSERT INTO vendors (vendor_name, supplier_no, contact_info, normalized_name, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (name, code, email, name.lower().strip()))
            
            vendor_id = cursor.lastrowid
            if not vendor_id:
                raise Exception("Failed to get vendor ID after insertion")
            
            conn.commit()
            
            debug_logger.info("Vendor inserted successfully", {
                "name": name, 
                "code": code, 
                "vendor_id": vendor_id
            })
            
            # Verify the insertion
            cursor.execute("SELECT vendor_name FROM vendors WHERE vendor_id = ?", (vendor_id,))
            verification = cursor.fetchone()
            if verification:
                debug_logger.debug("Vendor insertion verified", {
                    "vendor_id": vendor_id,
                    "stored_name": verification[0]
                })
                return vendor_id
            else:
                debug_logger.error("Vendor insertion verification failed", None, {
                    "vendor_id": vendor_id
                })
                raise Exception(f"Failed to verify vendor insertion")
                
    except Exception as e:
        debug_logger.exception("Error adding vendor to database", e, {
            "name": name,
            "code": code,
            "error_type": type(e).__name__
        })
        # Re-raise with more context
        if "UNIQUE constraint failed" in str(e):
            raise Exception(f"Vendor '{name}' already exists")
        else:
            raise Exception(f"Database error: {str(e)}")


def add_category(name: str, parent_id: Optional[int] = None, description: Optional[str] = None) -> int:
    """Add a new category to the database.
    
    Args:
        name: Category name (required)
        parent_id: Optional parent category ID
        
    Returns:
        int: The category_id of the newly created category
        
    Raises:
        Exception: If database operation fails
    """
    debug_logger.debug("Adding new category", {
        "name": name,
        "parent_id": parent_id
    })
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO categories (category_name, parent_category_id, description, created_at)
                VALUES (?, ?, ?, datetime('now'))
            """, (name, parent_id, description))
            conn.commit()
            category_id = cursor.lastrowid
            debug_logger.info("Category added successfully", {"name": name, "parent_id": parent_id})
            
            # Return category_id with null safety check
            return category_id if category_id is not None else 0
    except Exception as e:
        debug_logger.exception("Error adding category", e, {"name": name, "parent_id": parent_id})
        # Re-raise with more context
        if "UNIQUE constraint failed" in str(e):
            raise Exception(f"Category '{name}' already exists")
        else:
            raise Exception(f"Database error: {str(e)}")
