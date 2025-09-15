"""Vendor management page module."""
import streamlit as st
import pandas as pd
from typing import Optional
from src.utils.db_simple import get_db_connection
from src.utils.display import normalize_df_for_display
from src.utils.debug import debug_logger, show_error_block, safe_execute
from src.config import config
import shutil
import time


def render_page() -> None:
    """Render the vendor management page."""
    try:
        debug_logger.debug("Starting vendor management rendering")
        st.markdown("<div style='font-size:1.5em; color:#444; margin-bottom:18px;'> </div>", unsafe_allow_html=True)

    # Add new vendor form removed — use the editable table to add vendors instead

        # Display existing vendors. Provide toggle to show soft-deleted rows.
        success, vendors_df, error = safe_execute(
            load_vendors, False,
            error_title="Failed to Load Vendors",
            show_ui_error=True
        )
        
        # allow user to show soft-deleted rows
        try:
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.markdown(f"<div style='font-size:1.02em; color:#444; margin-top:10px;'>Existing Vendors ({len(vendors_df)})</div>", unsafe_allow_html=True)
            with col2:
                show_deleted = st.checkbox("Show soft-deleted rows", value=False, key="show_deleted_checkbox")
        except Exception:
            show_deleted = False
            
        if show_deleted:
            success, vendors_df, error = safe_execute(
                load_vendors, True,
                error_title="Failed to Load Vendors",
                show_ui_error=True
            )

            # Hard-delete controls when showing deleted rows
            try:
                soft_options = []
                soft_map = {}
                if success and vendors_df is not None and 'is_deleted' in vendors_df.columns:
                    for _, r in vendors_df.iterrows():
                        try:
                            isdel = r.get('is_deleted')
                            if pd.isna(isdel):
                                isdel = 0
                            try:
                                isdel_int = int(isdel)
                            except Exception:
                                isdel_int = 1 if str(isdel).lower() in ('true', '1', 'yes') else 0
                        except Exception:
                            isdel_int = 0

                        if isdel_int == 1:
                            vid = r.get('vendor_id')
                            name = r.get('vendor_name') or ''
                            label = f"{vid} — {name}"
                            soft_options.append(label)
                            soft_map[label] = vid

                if soft_options:
                    st.markdown("**Hard-delete soft-deleted vendors**")
                    to_remove = st.multiselect("Select soft-deleted vendors to permanently delete", options=soft_options, key='vendor_hard_delete_select')
                    if to_remove:
                        if st.button("Permanently delete selected"):
                            if not st.checkbox("Confirm permanent delete of selected items"):
                                st.info("Check confirmation to permanently delete selected items")
                            else:
                                db_path = config.database.path
                                timestamp = int(time.time())
                                bak = f"{db_path}.hard_delete_backup.{timestamp}.bak"
                                try:
                                    shutil.copyfile(db_path, bak)
                                except Exception as e:
                                    show_error_block("Failed to create DB backup", e)
                                    bak = None

                                try:
                                    ids = [soft_map[s] for s in to_remove]
                                    ids_int = []
                                    for v in ids:
                                        try:
                                            ids_int.append(int(v))
                                        except Exception:
                                            continue

                                    if ids_int:
                                        placeholders = ','.join(['?'] * len(ids_int))
                                        with get_db_connection() as conn:
                                            cur = conn.cursor()
                                            try:
                                                conn.execute('BEGIN')
                                            except Exception:
                                                pass
                                            cur.execute(f"DELETE FROM vendors WHERE vendor_id IN ({placeholders})", tuple(ids_int))
                                            conn.commit()

                                    st.success(f"Permanently deleted {len(ids_int)} vendors")
                                    rerun_fn = getattr(st, 'experimental_rerun', None)
                                    if callable(rerun_fn):
                                        try:
                                            rerun_fn()
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
                                            show_error_block("Failed to apply hard delete; DB restored from backup", e)
                                        except Exception as re:
                                            show_error_block("Failed to apply hard delete and failed to restore backup", re)
                                    else:
                                        show_error_block("Failed to apply hard delete", e)

                    if st.button("Permanently delete ALL soft-deleted vendors"):
                        if st.checkbox("Confirm permanent delete of ALL soft-deleted vendors (irreversible)"):
                            db_path = config.database.path
                            timestamp = int(time.time())
                            bak = f"{db_path}.hard_delete_backup.{timestamp}.bak"
                            try:
                                shutil.copyfile(db_path, bak)
                            except Exception as e:
                                show_error_block("Failed to create DB backup", e)
                                bak = None

                            try:
                                with get_db_connection() as conn:
                                    cur = conn.cursor()
                                    try:
                                        conn.execute('BEGIN')
                                    except Exception:
                                        pass
                                    cur.execute("DELETE FROM vendors WHERE COALESCE(is_deleted,0) = 1")
                                    conn.commit()
                                st.success("Permanently deleted all soft-deleted vendors")
                                rerun_fn = getattr(st, 'experimental_rerun', None)
                                if callable(rerun_fn):
                                    try:
                                        rerun_fn()
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
                                        show_error_block("Failed to delete all; DB restored from backup", e)
                                    except Exception as re:
                                        show_error_block("Failed to delete all and failed to restore backup", re)
                                else:
                                    show_error_block("Failed to delete all soft-deleted vendors", e)
                else:
                    st.info("No soft-deleted vendors found")
            except Exception:
                # non-fatal: continue without hard-delete UI
                pass

        if success and vendors_df is not None and not vendors_df.empty:
            #st.markdown(f"<div style='font-size:1.02em; color:#444; margin-bottom:18px;'> Existing Vendors ({len(vendors_df)}) </div>", unsafe_allow_html=True)
            
            # Normalize and prepare display dataframe
            try:
                display_df = normalize_df_for_display(vendors_df.copy())
            except Exception:
                display_df = vendors_df

            # Rely on Streamlit's editable data editor
            try:
                column_config = {}
                try:
                    if 'vendor_id' in display_df.columns:
                        column_config['vendor_id'] = st.column_config.Column(label='ID', disabled=True)
                except Exception:
                    column_config = {}

                # Prefer the last editor state if present, otherwise session override or display_df
                if 'vendors_data_editor' in st.session_state:
                    try:
                        df_to_edit = pd.DataFrame(st.session_state['vendors_data_editor'])
                    except Exception:
                        df_to_edit = st.session_state.get('vendors_data_editor_df', display_df)
                else:
                    df_to_edit = st.session_state.get('vendors_data_editor_df', display_df)

                # Move newly added rows (missing vendor_id) to the top so the add toolbar behaves like 'add at top'
                try:
                    if 'vendor_id' in df_to_edit.columns:
                        mask_new = df_to_edit['vendor_id'].isna()
                        if mask_new.any():
                            new_rows = df_to_edit[mask_new]
                            existing = df_to_edit[~mask_new]
                            df_to_edit = pd.concat([new_rows, existing], ignore_index=True, sort=False)
                            # persist ordering so editor shows new rows at top
                            st.session_state['vendors_data_editor_df'] = df_to_edit
                            try:
                                st.session_state['vendors_data_editor'] = df_to_edit.to_dict('records')
                            except Exception:
                                pass
                except Exception:
                    pass
                edited = st.data_editor(
                    df_to_edit,
                    width='stretch',
                    num_rows="dynamic",
                    column_config=(column_config if column_config else None),
                    key='vendors_data_editor'
                )

                # If user added rows via the editor, move new rows (null vendor_id) to top and persist
                try:
                    if isinstance(edited, pd.DataFrame) and 'vendor_id' in edited.columns:
                        new_mask = edited['vendor_id'].isna()
                        if new_mask.any():
                            reordered = pd.concat([edited[new_mask], edited[~new_mask]], ignore_index=True, sort=False)
                            st.session_state['vendors_data_editor'] = reordered.to_dict('records')
                            st.session_state['vendors_data_editor_df'] = reordered
                            rerun_fn = getattr(st, 'experimental_rerun', None)
                            if callable(rerun_fn):
                                try:
                                    rerun_fn()
                                except Exception:
                                    pass
                except Exception:
                    pass
            except Exception:
                edited = st.data_editor(display_df, width='stretch', num_rows="dynamic", key='vendors_data_editor')

            # Per-row validation
            validation_msgs = []
            try:
                for idx, row in edited.reset_index().iterrows():
                    row_id = row.get('vendor_id') if 'vendor_id' in row.index else None
                    name = row.get('vendor_name')
                    if name is None or str(name).strip() == '':
                        validation_msgs.append((idx, row_id, 'vendor_name is required'))
            except Exception:
                validation_msgs = []

            if validation_msgs:
                with st.expander(f"Validation issues ({len(validation_msgs)})"):
                    for idx, rid, msg in validation_msgs:
                        st.warning(f"Row #{idx} (id={rid}): {msg}")

            # Save changes
            if st.button("Save changes to vendors"):
                # Determine changes by vendor_id
                orig = display_df.set_index('vendor_id') if 'vendor_id' in display_df.columns else display_df
                new = edited.set_index('vendor_id') if 'vendor_id' in edited.columns else edited

                # Validation
                invalid_rows = []
                for _, row in new.reset_index().iterrows():
                    name = row.get('vendor_name')
                    if name is None or str(name).strip() == '':
                        invalid_rows.append(row.get('vendor_id'))

                if invalid_rows:
                    st.error(f"Validation failed: {len(invalid_rows)} row(s) have empty vendor_name. Please fill them before saving.")
                else:
                    # Compute deletes
                    try:
                        orig_ids = set()
                        if 'vendor_id' in display_df.columns:
                            for v in display_df['vendor_id']:
                                if pd.isna(v):
                                    continue
                                try:
                                    orig_ids.add(int(v))
                                except Exception:
                                    continue

                        new_ids = set()
                        if 'vendor_id' in edited.columns:
                            for v in edited['vendor_id']:
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
                        st.warning(f"This operation will SOFT-DELETE {len(to_delete)} existing vendor(s). This can be undone by restoring a backup.")
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
                                cur.execute("PRAGMA table_info(vendors)")
                                cols = [r[1] for r in cur.fetchall()]
                                has_is_deleted = 'is_deleted' in cols

                                # ensure is_deleted exists if possible
                                if 'is_deleted' not in cols:
                                    try:
                                        cur.execute("ALTER TABLE vendors ADD COLUMN is_deleted INTEGER DEFAULT 0")
                                        cur.execute("PRAGMA table_info(vendors)")
                                        cols = [r[1] for r in cur.fetchall()]
                                        has_is_deleted = 'is_deleted' in cols
                                    except Exception:
                                        pass

                                try:
                                    conn.execute('BEGIN')
                                except Exception:
                                    pass

                                for vid in to_delete:
                                    try:
                                        if 'is_deleted' in cols:
                                            cur.execute("UPDATE vendors SET is_deleted = 1 WHERE vendor_id = ?", (vid,))
                                        else:
                                            cur.execute("DELETE FROM vendors WHERE vendor_id = ?", (vid,))
                                    except Exception:
                                        raise

                                for _, row in edited.reset_index().iterrows():
                                    vid = row.get('vendor_id') if 'vendor_id' in row.index or 'vendor_id' in edited.columns else None
                                    name = row.get('vendor_name')
                                    supplier_no = row.get('supplier_no') if 'supplier_no' in row.index else None
                                    contact = row.get('contact_info') if 'contact_info' in row.index else None

                                    if pd.isna(vid) or vid is None:
                                        # insert
                                        try:
                                            cur.execute(
                                                "INSERT INTO vendors (vendor_name, supplier_no, contact_info, normalized_name, is_deleted, created_at) VALUES (?, ?, ?, ?, 0, datetime('now'))",
                                                (name, supplier_no, contact, str(name).lower().strip()),
                                            )
                                        except Exception:
                                            # fallback if is_deleted not present
                                            cur.execute(
                                                "INSERT INTO vendors (vendor_name, supplier_no, contact_info, normalized_name, created_at) VALUES (?, ?, ?, ?, datetime('now'))",
                                                (name, supplier_no, contact, str(name).lower().strip()),
                                            )
                                        inserted += 1
                                    else:
                                        try:
                                            vid_int = int(vid)
                                        except Exception:
                                            continue
                                        cur.execute(
                                            "UPDATE vendors SET vendor_name = ?, supplier_no = ?, contact_info = ?, normalized_name = ?, updated_at = datetime('now') WHERE vendor_id = ?",
                                            (name, supplier_no, contact, str(name).lower().strip(), vid_int),
                                        )
                                        applied += 1

                                conn.commit()

                            st.success(f"Applied {applied} updates, inserted {inserted} new vendors")
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

        elif success:
            st.info("No vendors found. Add some vendors to get started.")

    except Exception as e:
        debug_logger.exception("Error in vendor management rendering", e)
        show_error_block("Vendor Management Error", e)


def load_vendors(show_deleted: bool = False) -> pd.DataFrame:
    """Load vendors from database.

    If show_deleted is False and the vendors table has an `is_deleted` column,
    only return rows where is_deleted is 0 (or NULL treated as 0).
    """
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(vendors)")
            cols = [r[1] for r in cur.fetchall()]
            has_is_deleted = 'is_deleted' in cols

            base = "SELECT vendor_id, vendor_name, supplier_no, contact_info, created_at, updated_at FROM vendors"
            if not show_deleted and has_is_deleted:
                query = base + " WHERE COALESCE(is_deleted, 0) = 0 ORDER BY vendor_name"
            else:
                query = base + " ORDER BY vendor_name"

            df = pd.read_sql_query(query, conn)
            return df
    except Exception as e:
        debug_logger.exception("Error loading vendors", e)
        raise Exception(f"Database error loading vendors: {str(e)}")


def add_vendor(name: str, code: Optional[str] = None, email: Optional[str] = None) -> int:
    """Add a new vendor to the database."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT vendor_id FROM vendors WHERE normalized_name = ?", (name.lower().strip(),))
            existing = cursor.fetchone()
            if existing:
                raise Exception(f"Vendor '{name}' already exists")

            cursor.execute(
                """
                INSERT INTO vendors (vendor_name, supplier_no, contact_info, normalized_name, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (name, code, email, name.lower().strip()),
            )
            vendor_id = cursor.lastrowid
            conn.commit()
            return int(vendor_id) if vendor_id is not None else 0
    except Exception as e:
        debug_logger.exception("Error adding vendor to database", e, {"name": name, "code": code})
        if "UNIQUE constraint failed" in str(e):
            raise Exception(f"Vendor '{name}' already exists")
        else:
            raise Exception(f"Database error: {str(e)}")
