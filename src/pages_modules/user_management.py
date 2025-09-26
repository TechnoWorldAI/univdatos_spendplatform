"""User management page module."""
import streamlit as st
import pandas as pd
from src.utils.db_simple import get_db_connection
from src.services.auth_service import AuthService
from src.exceptions.base import ValidationError
from src.config import config
from src.utils.debug import safe_execute, show_error_block, debug_logger
from src.utils.display import normalize_df_for_display
import shutil
import time


def render_page() -> None:
    """Render the user management page."""
    st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> 👥 User Management</div>", unsafe_allow_html=True)
    
    # Check permissions
    user_role = st.session_state.user.get('role', '')
    if user_role != 'Admin':
        st.error("❌ Access denied. Admin role required.")
        return
    
    # Load users (use safe_execute to handle DB errors)
    success, users_df, error = safe_execute(load_users, False, error_title="Failed to Load Users", show_ui_error=True)
    if not success:
        # safe_execute already showed an error block if requested; stop rendering
        return
    
    # Add new user form removed — use the editable table to add users instead
    
    # Display existing users in an editable grid
    if success and users_df is not None and not users_df.empty:


        # Prepare display dataframe — guard against None
        if users_df is None:
            display_df = pd.DataFrame()
        else:
            try:
                display_df = normalize_df_for_display(users_df.copy())
            except Exception:
                display_df = users_df.copy() if isinstance(users_df, pd.DataFrame) else pd.DataFrame()

        # Use editable data editor
        try:
            column_config = {}
            try:
                if isinstance(display_df, pd.DataFrame) and 'user_id' in display_df.columns:
                    column_config['user_id'] = st.column_config.Column(label='ID', disabled=True)
            except Exception:
                column_config = {}

            # Prefer last editor state
            if 'users_data_editor' in st.session_state:
                try:
                    df_to_edit = pd.DataFrame(st.session_state['users_data_editor'])
                except Exception:
                    df_to_edit = st.session_state.get('users_data_editor_df', display_df)
            else:
                df_to_edit = st.session_state.get('users_data_editor_df', display_df)

            # Move newly added rows to top
            try:
                if isinstance(df_to_edit, pd.DataFrame) and 'user_id' in df_to_edit.columns:
                    mask_new = df_to_edit['user_id'].isna()
                    if mask_new.any():
                        new_rows = df_to_edit[mask_new]
                        existing = df_to_edit[~mask_new]
                        df_to_edit = pd.concat([new_rows, existing], ignore_index=True, sort=False)
                        st.session_state['users_data_editor_df'] = df_to_edit
                        try:
                            st.session_state['users_data_editor'] = df_to_edit.to_dict('records')
                        except Exception:
                            pass
            except Exception:
                pass

            edited = st.data_editor(
                df_to_edit,
                width='stretch',
                num_rows="dynamic",
                column_config=(column_config if column_config else None),
                key='users_data_editor'
            )

            # If user added rows via the editor, move new rows (null user_id) to top and persist
            try:
                if isinstance(edited, pd.DataFrame) and 'user_id' in edited.columns:
                    new_mask = edited['user_id'].isna()
                    if new_mask.any():
                        reordered = pd.concat([edited[new_mask], edited[~new_mask]], ignore_index=True, sort=False)
                        st.session_state['users_data_editor'] = reordered.to_dict('records')
                        st.session_state['users_data_editor_df'] = reordered
                        rerun_fn = getattr(st, 'experimental_rerun', None)
                        if callable(rerun_fn):
                            try:
                                rerun_fn()
                            except Exception:
                                pass
            except Exception:
                pass
        except Exception:
            edited = st.data_editor(display_df, width='stretch', num_rows="dynamic", key='users_data_editor')

        # Validation
        validation_msgs = []
        try:
            for idx, row in edited.reset_index().iterrows():
                row_id = row.get('user_id') if 'user_id' in row.index else None
                uname = row.get('username')
                if uname is None or str(uname).strip() == '':
                    validation_msgs.append((idx, row_id, 'username is required'))
        except Exception:
            validation_msgs = []

        if validation_msgs:
            with st.expander(f"Validation issues ({len(validation_msgs)})"):
                for idx, rid, msg in validation_msgs:
                    st.warning(f"Row #{idx} (id={rid}): {msg}")

        # Save changes
        if st.button("Save changes to users"):
            orig = display_df.set_index('user_id') if (isinstance(display_df, pd.DataFrame) and 'user_id' in display_df.columns) else display_df
            new = edited.set_index('user_id') if 'user_id' in edited.columns else edited

            # Validate
            invalid_rows = []
            for _, row in new.reset_index().iterrows():
                uname = row.get('username')
                if uname is None or str(uname).strip() == '':
                    invalid_rows.append(row.get('user_id'))

            if invalid_rows:
                st.error(f"Validation failed: {len(invalid_rows)} row(s) have empty username. Please fill them before saving.")
            else:
                try:
                    orig_ids = set()
                    if isinstance(display_df, pd.DataFrame) and 'user_id' in display_df.columns:
                        for v in display_df['user_id']:
                            if pd.isna(v):
                                continue
                            try:
                                orig_ids.add(int(v))
                            except Exception:
                                continue

                    new_ids = set()
                    if 'user_id' in edited.columns:
                        for v in edited['user_id']:
                            if pd.isna(v):
                                continue
                            try:
                                new_ids.add(int(v))
                            except Exception:
                                continue

                    to_delete = sorted(list(orig_ids - new_ids))
                except Exception:
                    to_delete = []

                if to_delete:
                    st.warning(f"This operation will SOFT-DELETE {len(to_delete)} existing user(s). This can be undone by restoring a backup.")
                    if not st.button("Confirm soft-delete and apply changes"):
                        st.info("Click 'Confirm soft-delete and apply changes' to proceed.")
                        pass
                    else:
                        apply_now = True
                else:
                    apply_now = True

                if 'apply_now' in locals() and apply_now:
                    db_path = config.database.path
                    timestamp = int(time.time())
                    bak = f"{db_path}.edit_backup.{timestamp}.bak"
                    try:
                        shutil.copyfile(db_path, bak)
                    except Exception as e:
                        show_error_block("Failed to create DB backup", e)
                        bak = None

                    applied = 0
                    inserted = 0
                    try:
                        with get_db_connection() as conn:
                            cur = conn.cursor()
                            cur.execute("PRAGMA table_info(users)")
                            cols = [r[1] for r in cur.fetchall()]
                            has_is_deleted = 'is_deleted' in cols

                            if 'is_deleted' not in cols:
                                try:
                                    cur.execute("ALTER TABLE users ADD COLUMN is_deleted INTEGER DEFAULT 0")
                                    cur.execute("PRAGMA table_info(users)")
                                    cols = [r[1] for r in cur.fetchall()]
                                except Exception:
                                    pass

                            try:
                                conn.execute('BEGIN')
                            except Exception:
                                pass

                            for uid in to_delete:
                                try:
                                    if 'is_deleted' in cols:
                                        cur.execute("UPDATE users SET is_deleted = 1 WHERE user_id = ?", (uid,))
                                    else:
                                        cur.execute("DELETE FROM users WHERE user_id = ?", (uid,))
                                except Exception:
                                    raise

                            auth_service = AuthService()
                            for _, row in edited.reset_index().iterrows():
                                uid = row.get('user_id') if 'user_id' in row.index or 'user_id' in edited.columns else None
                                uname = row.get('username')
                                role = row.get('role') if 'role' in row.index else None
                                new_password = row.get('new_password') if 'new_password' in row.index else None

                                if pd.isna(uid) or uid is None:
                                    # create user via service to handle hashing/validation
                                    try:
                                        auth_service.create_user(str(uname).strip(), str(new_password) if new_password else '', str(role) if role else 'Spend Manager')
                                        inserted += 1
                                    except ValidationError as e:
                                        raise
                                else:
                                    try:
                                        uid_int = int(uid)
                                    except Exception:
                                        continue
                                    updates = {'username': str(uname).strip()}
                                    if role:
                                        updates['role'] = role
                                    if new_password and str(new_password).strip() != '':
                                        updates['password'] = str(new_password)
                                    try:
                                        auth_service.update_user(uid_int, updates)
                                        applied += 1
                                    except ValidationError:
                                        raise

                            conn.commit()

                        st.success(f"Applied {applied} updates, inserted {inserted} new users")
                        rerun_fn = getattr(st, 'experimental_rerun', None)
                        if callable(rerun_fn):
                            try:
                                rerun_fn()
                            except Exception:
                                pass
                        else:
                            try:
                                st.rerun()
                            except Exception:
                                pass

                    except Exception as e:
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        if bak:
                            try:
                                shutil.copyfile(bak, db_path)
                                show_error_block("Failed to apply changes; DB restored from backup", e)
                            except Exception as re:
                                show_error_block("Failed to apply changes and failed to restore backup", re)
                        else:
                            show_error_block("Failed to apply changes", e)
    else:
        st.info("No users found.")


def load_users(show_deleted: bool = False) -> pd.DataFrame:
    """Load user data from database.

    If show_deleted is False and the users table has an `is_deleted` column,
    only return rows where is_deleted is 0 (or NULL treated as 0).
    """
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(users)")
            cols = [r[1] for r in cur.fetchall()]
            has_is_deleted = 'is_deleted' in cols

            base = "SELECT * FROM users"
            if not show_deleted and has_is_deleted:
                query = base + " WHERE COALESCE(is_deleted, 0) = 0 ORDER BY username"
            else:
                query = base + " ORDER BY username"

            return pd.read_sql_query(query, conn)
    except Exception as e:
        debug_logger.exception("Error loading users", e)
        raise Exception(f"Database error loading users: {str(e)}")


def delete_user(user_id: int) -> bool:
    """Delete a user from the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception:
        return False
