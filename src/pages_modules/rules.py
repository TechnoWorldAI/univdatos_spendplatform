"""Rules management page module."""
import streamlit as st
import pandas as pd
from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block, safe_execute
from src.utils.display import normalize_df_for_display
from src.config import config
import shutil
import time


def render_page() -> None:
    """Render the rules management page."""
    try:
        debug_logger.info("Rendering rules management page")
        st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> ⚙️ Rules Management</div>", unsafe_allow_html=True)

        # Permission check: allow Admin and Spend Manager
        user_role = st.session_state.get('user', {}).get('role', '')
        if user_role not in ("Admin", "Spend Manager"):
            st.error("❌ Access denied. Admin or Spend Manager role required.")
            return

        # Use an editable table for rules; add via the table toolbar
        render_rule_list()

    except Exception as e:
        debug_logger.exception("Error rendering rules page", e)
        show_error_block("Rules Page Error", e)


def render_add_rule():
    # kept for compatibility but not used; rules are added inline in the table
    st.info("Add rule via the editable table on the left")


def render_rule_list():
    """Display and edit existing rules using an editable grid."""

    success, df, error = safe_execute(load_rules, False, error_title="Failed to Load Rules", show_ui_error=True)
    if not success or df is None:
        return

    if df.empty:
        st.info("No rules found. Add a rule using the editable table.")
        return

    # show_deleted toggle
    try:
        col1, col2 = st.columns([0.7, 0.3])
        with col1:
            st.markdown(f"<div style='font-size:1.02em; color:#444; margin-top:10px;'>Existing Rules ({len(df)})</div>", unsafe_allow_html=True)
        with col2:
            show_deleted = st.checkbox("Show soft-deleted rows", value=False, key="show_deleted_checkbox")
    except Exception:
            show_deleted = False

    if show_deleted:
        success, df, error = safe_execute(load_rules, True, error_title="Failed to Load Rules", show_ui_error=True)
        try:
            soft_options = []
            soft_map = {}
            if df is not None and 'is_deleted' in df.columns:
                for _, r in df.iterrows():
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
                        rid = r.get('rule_id')
                        name = r.get('rule_name') or ''
                        label = f"{rid} — {name}"
                        soft_options.append(label)
                        soft_map[label] = rid

            if soft_options:
                st.markdown("**Hard-delete soft-deleted rules**")
                to_remove = st.multiselect("Select soft-deleted rules to permanently delete", options=soft_options, key='rule_hard_delete_select')
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
                                        cur.execute(f"DELETE FROM rules WHERE rule_id IN ({placeholders})", tuple(ids_int))
                                        conn.commit()

                                st.success(f"Permanently deleted {len(ids_int)} rules")
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

                if st.button("Permanently delete ALL soft-deleted rules"):
                    if st.checkbox("Confirm permanent delete of ALL soft-deleted rules (irreversible)"):
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
                                cur.execute("DELETE FROM rules WHERE COALESCE(is_deleted,0) = 1")
                                conn.commit()
                            st.success("Permanently deleted all soft-deleted rules")
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
                                show_error_block("Failed to delete all soft-deleted rules", e)
            else:
                st.info("No soft-deleted rules found")
        except Exception:
            pass

    # Normalize for display
    if df is None:
            display_df = pd.DataFrame()
    else:
        try:
            display_df = normalize_df_for_display(df.copy())
        except Exception:
            display_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    # Editable data editor
    try:
        column_config = {}
        try:
            if isinstance(display_df, pd.DataFrame) and 'rule_id' in display_df.columns:
                column_config['rule_id'] = st.column_config.Column(label='ID', disabled=True)
        except Exception:
            column_config = {}

        # session override
        if 'rules_data_editor' in st.session_state:
            try:
                df_to_edit = pd.DataFrame(st.session_state['rules_data_editor'])
            except Exception:
                df_to_edit = st.session_state.get('rules_data_editor_df', display_df)
        else:
            df_to_edit = st.session_state.get('rules_data_editor_df', display_df)

        # Move new rows to top
        try:
            if isinstance(df_to_edit, pd.DataFrame) and 'rule_id' in df_to_edit.columns:
                mask_new = df_to_edit['rule_id'].isna()
                if mask_new.any():
                    new_rows = df_to_edit[mask_new]
                    existing = df_to_edit[~mask_new]
                    df_to_edit = pd.concat([new_rows, existing], ignore_index=True, sort=False)
                    # persist both df and the raw editor data so Streamlit shows the ordering immediately
                    st.session_state['rules_data_editor_df'] = df_to_edit
                    try:
                        st.session_state['rules_data_editor'] = df_to_edit.to_dict('records')
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            edited = st.data_editor(
                df_to_edit,
                width='stretch',
                num_rows="dynamic",
                column_config=(column_config if column_config else None),
                key='rules_data_editor'
            )
        except Exception:
            edited = st.data_editor(display_df, width='stretch', num_rows="dynamic", key='rules_data_editor')

        # If user added rows via the editor, move new rows (null rule_id) to top and persist
        try:
            if isinstance(edited, pd.DataFrame) and 'rule_id' in edited.columns:
                new_mask = edited['rule_id'].isna()
                if new_mask.any():
                    reordered = pd.concat([edited[new_mask], edited[~new_mask]], ignore_index=True, sort=False)
                    st.session_state['rules_data_editor'] = reordered.to_dict('records')
                    st.session_state['rules_data_editor_df'] = reordered
                    rerun_fn = getattr(st, 'experimental_rerun', None)
                    if callable(rerun_fn):
                        try:
                            rerun_fn()
                        except Exception:
                            pass
        except Exception:
            pass

    except Exception:
        # If the whole editable editor block fails, fall back to a safe empty/readonly edited
        try:
            edited = display_df.copy() if isinstance(display_df, pd.DataFrame) else pd.DataFrame()
        except Exception:
            edited = pd.DataFrame()
        try:
            # clear any problematic session override
            st.session_state.pop('rules_data_editor', None)
        except Exception:
            pass

    # Validation
    validation_msgs = []
    try:
        for idx, row in edited.reset_index().iterrows():
            rid = row.get('rule_id') if 'rule_id' in row.index else None
            name = row.get('rule_name')
            if name is None or str(name).strip() == '':
                validation_msgs.append((idx, rid, 'rule_name is required'))
    except Exception:
        validation_msgs = []

    if validation_msgs:
        with st.expander(f"Validation issues ({len(validation_msgs)})"):
            for idx, rid, msg in validation_msgs:
                st.warning(f"Row #{idx} (id={rid}): {msg}")

    # Save changes
    if st.button("Save changes to rules"):
        orig = display_df.set_index('rule_id') if (isinstance(display_df, pd.DataFrame) and 'rule_id' in display_df.columns) else display_df
        new = edited.set_index('rule_id') if 'rule_id' in edited.columns else edited

        # Validate basic required fields
        invalid_rows = []
        for _, row in new.reset_index().iterrows():
            name = row.get('rule_name')
            if name is None or str(name).strip() == '':
                invalid_rows.append(row.get('rule_id'))

        if invalid_rows:
            st.error(f"Validation failed: {len(invalid_rows)} row(s) have empty rule_name. Please fill them before saving.")
        else:
            try:
                orig_ids = set()
                if isinstance(display_df, pd.DataFrame) and 'rule_id' in display_df.columns:
                    for v in display_df['rule_id']:
                        if pd.isna(v):
                            continue
                        try:
                            orig_ids.add(int(v))
                        except Exception:
                            continue

                new_ids = set()
                if 'rule_id' in edited.columns:
                    for v in edited['rule_id']:
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
                st.warning(f"This operation will SOFT-DELETE {len(to_delete)} existing rule(s). This can be undone by restoring a backup.")
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
                        cur.execute("PRAGMA table_info(rules)")
                        cols = [r[1] for r in cur.fetchall()]

                        if 'is_deleted' not in cols:
                            try:
                                cur.execute("ALTER TABLE rules ADD COLUMN is_deleted INTEGER DEFAULT 0")
                                cur.execute("PRAGMA table_info(rules)")
                                cols = [r[1] for r in cur.fetchall()]
                            except Exception:
                                pass

                        try:
                            conn.execute('BEGIN')
                        except Exception:
                            pass

                        for rid in to_delete:
                            try:
                                if 'is_deleted' in cols:
                                    cur.execute("UPDATE rules SET is_deleted = 1 WHERE rule_id = ?", (rid,))
                                else:
                                    cur.execute("DELETE FROM rules WHERE rule_id = ?", (rid,))
                            except Exception:
                                raise

                        for _, row in edited.reset_index().iterrows():
                            rid = row.get('rule_id') if 'rule_id' in row.index or 'rule_id' in edited.columns else None
                            name = row.get('rule_name')
                            desc = row.get('rule_description') if 'rule_description' in row.index else None
                            rtype = row.get('rule_type') if 'rule_type' in row.index else None
                            cond = row.get('rule_condition') if 'rule_condition' in row.index else None
                            active = 1 if row.get('active_flag') else 0

                            if pd.isna(rid) or rid is None:
                                # insert
                                if 'is_deleted' in cols:
                                    cur.execute(
                                        "INSERT INTO rules (rule_name, rule_description, rule_type, rule_condition, active_flag, is_deleted, created_at) VALUES (?, ?, ?, ?, ?, 0, datetime('now'))",
                                        (name, desc, rtype, cond, active),
                                    )
                                else:
                                    cur.execute(
                                        "INSERT INTO rules (rule_name, rule_description, rule_type, rule_condition, active_flag, created_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                                        (name, desc, rtype, cond, active),
                                    )
                                inserted += 1
                            else:
                                try:
                                    rid_int = int(rid)
                                except Exception:
                                    continue
                                cur.execute(
                                    "UPDATE rules SET rule_name = ?, rule_description = ?, rule_type = ?, rule_condition = ?, active_flag = ? WHERE rule_id = ?",
                                    (name, desc, rtype, cond, active, rid_int),
                                )
                                applied += 1

                        conn.commit()

                    st.success(f"Applied {applied} updates, inserted {inserted} new rules")
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


def load_rules(show_deleted: bool = False) -> pd.DataFrame:
    """Load rules from the database.

    If show_deleted is False and the rules table has an `is_deleted` column,
    only return rows where is_deleted is 0 (or NULL treated as 0).
    """
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(rules)")
            cols = [r[1] for r in cur.fetchall()]
            has_is_deleted = 'is_deleted' in cols

            base = "SELECT * FROM rules"
            if not show_deleted and has_is_deleted:
                query = base + " WHERE COALESCE(is_deleted, 0) = 0 ORDER BY created_at DESC"
            else:
                query = base + " ORDER BY created_at DESC"

            return pd.read_sql_query(query, conn)
    except Exception as e:
        debug_logger.exception("Error loading rules", e)
        raise Exception(f"Database error loading rules: {str(e)}")


def update_rule(rule_id: int, name: str, description: str, rule_type: str, condition: str, active_flag: int) -> None:
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE rules SET rule_name = ?, rule_description = ?, rule_type = ?, rule_condition = ?, active_flag = ? WHERE rule_id = ?",
            (name, description, rule_type, condition, active_flag, rule_id)
        )
        conn.commit()


def delete_rule(rule_id: int) -> bool:
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM rules WHERE rule_id = ?", (rule_id,))
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        debug_logger.exception("Failed to delete rule", e)
        return False
