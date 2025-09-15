"""Base manager class for CRUD operations with common functionality."""
import pandas as pd
import streamlit as st
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import time
import shutil

from src.utils.db_simple import get_db_connection
from src.utils.debug import debug_logger, show_error_block
from src.utils.display import normalize_df_for_display
from src.config import config


class BaseManager(ABC):
    """Base class for managing CRUD operations with common patterns."""

    def __init__(self, table_name: str, primary_key: str, display_name: str):
        self.table_name = table_name
        self.primary_key = primary_key
        self.display_name = display_name

    @abstractmethod
    def get_column_config(self) -> Dict[str, Any]:
        """Return column configuration for data_editor."""
        pass

    @abstractmethod
    def validate_row(self, row: pd.Series) -> List[str]:
        """Validate a single row and return list of error messages."""
        pass

    def load_data(self, show_deleted: bool = False) -> pd.DataFrame:
        """Load data from database with optional soft-deleted records."""
        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                
                # Safe table info query
                self._execute_safe_query(cur, "PRAGMA table_info({table})", 
                                       table_name=self.table_name)
                cols = [r[1] for r in cur.fetchall()]
                has_is_deleted = 'is_deleted' in cols

                if not show_deleted and has_is_deleted:
                    query = "SELECT * FROM {table} WHERE COALESCE(is_deleted, 0) = 0"
                else:
                    query = "SELECT * FROM {table}"
                    
                self._execute_safe_query(cur, query, table_name=self.table_name)
                df = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
                return normalize_df_for_display(df)
        except Exception as e:
            debug_logger.exception(f"Error loading {self.display_name}", e)
            raise Exception(f"Database error loading {self.display_name}: {str(e)}")

    def save_changes(self, original_df: pd.DataFrame, edited_df: pd.DataFrame) -> Tuple[bool, str]:
        """Save changes with backup and transaction support."""
        # Create backup
        db_path = config.database.path
        timestamp = int(time.time())
        backup_path = f"{db_path}.edit_backup.{timestamp}.bak"

        try:
            shutil.copyfile(db_path, backup_path)
        except Exception as e:
            return False, f"Failed to create backup: {str(e)}"

        try:
            with get_db_connection() as conn:
                cur = conn.cursor()

                # Ensure is_deleted column exists
                self._execute_safe_query(cur, "PRAGMA table_info({table})", 
                                       table_name=self.table_name)
                cols = [r[1] for r in cur.fetchall()]
                if 'is_deleted' not in cols:
                    self._execute_safe_query(cur, "ALTER TABLE {table} ADD COLUMN is_deleted INTEGER DEFAULT 0",
                                           table_name=self.table_name)
                    cols.append('is_deleted')

                # Begin transaction
                conn.execute('BEGIN')

                # Calculate changes
                to_delete = self._calculate_deletions(original_df, edited_df)
                to_insert, to_update = self._calculate_inserts_updates(original_df, edited_df)

                # Apply soft deletes
                for item_id in to_delete:
                    if 'is_deleted' in cols:
                        self._execute_safe_query(cur, 
                                               "UPDATE {table} SET is_deleted = 1 WHERE {column} = ?",
                                               table_name=self.table_name, 
                                               column_name=self.primary_key,
                                               params=(item_id,))
                    else:
                        self._execute_safe_query(cur,
                                               "DELETE FROM {table} WHERE {column} = ?",
                                               table_name=self.table_name,
                                               column_name=self.primary_key, 
                                               params=(item_id,))

                # Apply inserts and updates
                for row in to_insert:
                    self._insert_row(cur, row, cols)
                for row in to_update:
                    self._update_row(cur, row, cols)

                conn.commit()
                return True, f"Successfully saved {len(to_insert)} inserts and {len(to_update)} updates"

        except Exception as e:
            # Rollback and restore backup
            try:
                conn.rollback()
            except Exception:
                pass

            try:
                shutil.copyfile(backup_path, db_path)
                return False, f"Failed to save changes; database restored from backup: {str(e)}"
            except Exception as restore_e:
                return False, f"Failed to save changes and failed to restore backup: {str(e)} / {str(restore_e)}"

    def _calculate_deletions(self, original_df: pd.DataFrame, edited_df: pd.DataFrame) -> List[int]:
        """Calculate which items were deleted."""
        if self.primary_key not in original_df.columns or self.primary_key not in edited_df.columns:
            return []

        orig_ids = set()
        for v in original_df[self.primary_key]:
            if pd.isna(v):
                continue
            try:
                orig_ids.add(int(v))
            except Exception:
                continue

        new_ids = set()
        for v in edited_df[self.primary_key]:
            if pd.isna(v):
                continue
            try:
                new_ids.add(int(v))
            except Exception:
                continue

        return sorted(list(orig_ids - new_ids))

    def _calculate_inserts_updates(self, original_df: pd.DataFrame, edited_df: pd.DataFrame) -> Tuple[List[pd.Series], List[pd.Series]]:
        """Calculate inserts and updates."""
        inserts = []
        updates = []

        for _, row in edited_df.iterrows():
            item_id = row.get(self.primary_key)
            if pd.isna(item_id):
                inserts.append(row)
            else:
                updates.append(row)

        return inserts, updates

    @abstractmethod
    def _insert_row(self, cursor, row: pd.Series, columns: List[str]) -> None:
        """Insert a new row."""
        pass

    @abstractmethod
    def _update_row(self, cursor, row: pd.Series, columns: List[str]) -> None:
        """Update an existing row."""
        pass

    def move_new_rows_to_top(self, df: pd.DataFrame) -> pd.DataFrame:
        """Move newly added rows (NaN primary key) to the top."""
        if self.primary_key not in df.columns:
            return df

        mask_new = df[self.primary_key].isna()
        if not mask_new.any():
            return df

        new_rows = df[mask_new]
        existing_rows = df[~mask_new]
        return pd.concat([new_rows, existing_rows], ignore_index=True, sort=False)

    def render_management_page(self) -> None:
        """Render the complete management page with common UI patterns."""
        try:
            st.title(f"📋 {self.display_name} Management")
            st.markdown(f"### Manage {self.display_name.lower()} records")

            # Load data
            success, df_result, error = self.safe_execute(self.load_data, False)
            if not success or df_result is None:
                return

            # Ensure df is a DataFrame
            df: pd.DataFrame = df_result if isinstance(df_result, pd.DataFrame) else pd.DataFrame()

            if df.empty:
                st.info(f"No {self.display_name.lower()} found. Add some records using the editable table.")
                df = pd.DataFrame()  # Ensure we have a DataFrame for the editor

            # Show deleted toggle
            show_deleted = st.checkbox("Show soft-deleted rows", value=False)
            if show_deleted:
                success, df_result, error = self.safe_execute(self.load_data, True)
                if success and df_result is not None and isinstance(df_result, pd.DataFrame):
                    df = df_result
                    self._render_hard_delete_ui(df)

            # Data editor
            if not df.empty:
                edited_df = self._render_data_editor(df)
                if edited_df is None or not isinstance(edited_df, pd.DataFrame):
                    edited_df = df

                # Validation
                validation_errors = self._validate_all_rows(edited_df)
                if validation_errors:
                    self._render_validation_errors(validation_errors)

                # Save button
                if st.button(f"Save changes to {self.display_name.lower()}"):
                    if validation_errors:
                        st.error("Please fix validation errors before saving.")
                    else:
                        success, message = self.save_changes(df, edited_df)
                        if success:
                            st.success(message)
                            self._trigger_rerun()
                        else:
                            st.error(message)

        except Exception as e:
            debug_logger.exception(f"Error rendering {self.display_name} management page", e)
            show_error_block(f"{self.display_name} Management Error", e)

    def _render_data_editor(self, df: pd.DataFrame) -> pd.DataFrame:
        """Render the data editor with session state persistence."""
        # Get session state key
        session_key = f"{self.table_name}_data_editor"
        df_session_key = f"{session_key}_df"

        # Clear any problematic session state for the widget key
        if session_key in st.session_state:
            existing_state = st.session_state[session_key]
            # If it's a list (from our previous to_dict('records')), clear it
            if isinstance(existing_state, list):
                del st.session_state[session_key]

        # Use session state if available
        if df_session_key in st.session_state:
            df_to_edit_raw = st.session_state[df_session_key]
            df_to_edit = df_to_edit_raw if isinstance(df_to_edit_raw, pd.DataFrame) else df
        else:
            df_to_edit = df

        # Move new rows to top
        df_to_edit = self.move_new_rows_to_top(df_to_edit)

        # Update session state for DataFrame only
        st.session_state[df_session_key] = df_to_edit

        # Render editor
        try:
            # Filter column config to only include columns that exist in the DataFrame
            full_config = self.get_column_config()
            available_columns = set(df_to_edit.columns)
            filtered_config = {k: v for k, v in full_config.items() if k in available_columns}

            edited = st.data_editor(
                df_to_edit,
                width='stretch',
                num_rows="dynamic",
                column_config=filtered_config,
                key=session_key
            )

            # If new rows were added, move them to top and rerun
            if isinstance(edited, pd.DataFrame) and self.primary_key in edited.columns:
                new_mask = edited[self.primary_key].isna()
                if new_mask.any():
                    reordered = self.move_new_rows_to_top(edited)
                    st.session_state[df_session_key] = reordered
                    self._trigger_rerun()

            return edited if isinstance(edited, pd.DataFrame) else df

        except Exception as e:
            debug_logger.exception("Error rendering data editor", e)
            return df

    def _render_hard_delete_ui(self, df: pd.DataFrame) -> None:
        """Render hard delete UI for soft-deleted records."""
        if 'is_deleted' not in df.columns:
            return

        soft_deleted = df[df['is_deleted'] == 1]
        if soft_deleted.empty:
            st.info(f"No soft-deleted {self.display_name.lower()} found")
            return

        soft_options = []
        soft_map = {}
        for _, row in soft_deleted.iterrows():
            item_id = row.get(self.primary_key)
            name = row.get(f"{self.table_name[:-1]}_name") or row.get('name') or str(item_id)
            label = f"{item_id} — {name}"
            soft_options.append(label)
            soft_map[label] = item_id

        st.markdown(f"**Hard-delete soft-deleted {self.display_name.lower()}**")
        to_remove = st.multiselect(
            f"Select soft-deleted {self.display_name.lower()} to permanently delete",
            options=soft_options,
            key=f"{self.table_name}_hard_delete_select"
        )

        if to_remove:
            if st.button("Permanently delete selected"):
                if st.checkbox("Confirm permanent delete of selected items"):
                    self._perform_hard_delete([soft_map[s] for s in to_remove])

    def _perform_hard_delete(self, item_ids: List[int]) -> None:
        """Perform hard delete of specified items."""
        db_path = config.database.path
        timestamp = int(time.time())
        backup_path = f"{db_path}.hard_delete_backup.{timestamp}.bak"

        try:
            shutil.copyfile(db_path, backup_path)
        except Exception as e:
            show_error_block("Failed to create backup", e)
            return

        try:
            with get_db_connection() as conn:
                cur = conn.cursor()
                conn.execute('BEGIN')

                placeholders = ','.join(['?'] * len(item_ids))
                query = f"DELETE FROM {{table}} WHERE {{column}} IN ({placeholders})"
                self._execute_safe_query(cur, query,
                                       table_name=self.table_name,
                                       column_name=self.primary_key,
                                       params=tuple(item_ids))
                conn.commit()

                st.success(f"Permanently deleted {len(item_ids)} {self.display_name.lower()}")
                self._trigger_rerun()

        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass

            try:
                shutil.copyfile(backup_path, db_path)
                show_error_block(f"Failed to hard delete; database restored from backup", e)
            except Exception as restore_e:
                show_error_block(f"Failed to hard delete and failed to restore backup", restore_e)

    def _validate_all_rows(self, df: pd.DataFrame) -> List[Tuple[int, Any, str]]:
        """Validate all rows and return list of (row_index, item_id, error_message)."""
        errors = []
        for idx, row in df.iterrows():
            row_errors = self.validate_row(row)
            item_id = row.get(self.primary_key)
            for error in row_errors:
                errors.append((idx, item_id, error))
        return errors

    def _render_validation_errors(self, errors: List[Tuple[int, Any, str]]) -> None:
        """Render validation errors in an expander."""
        with st.expander(f"Validation issues ({len(errors)})"):
            for idx, item_id, msg in errors:
                st.warning(f"Row #{idx} (id={item_id}): {msg}")

    def _trigger_rerun(self) -> None:
        """Trigger a Streamlit rerun."""
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

    def _execute_safe_query(self, cursor, query_template: str, table_name: Optional[str] = None, 
                           column_name: Optional[str] = None, params: tuple = ()) -> None:
        """Execute a parameterized query safely with table/column name validation.
        
        Args:
            cursor: Database cursor
            query_template: SQL template with placeholders
            table_name: Table name to substitute
            column_name: Column name to substitute  
            params: Query parameters
        """
        # Validate table and column names to prevent SQL injection
        allowed_tables = {
            'users', 'vendors', 'categories', 'spend_transactions', 
            'error_logs', 'rules', 'user_settings'
        }
        
        if table_name and table_name not in allowed_tables:
            raise ValueError(f"Invalid table name: {table_name}")
            
        if column_name and not column_name.replace('_', '').isalnum():
            raise ValueError(f"Invalid column name: {column_name}")
            
        # Safe substitution
        query = query_template
        if table_name:
            query = query.replace('{table}', table_name)
        if column_name:
            query = query.replace('{column}', column_name)
            
        cursor.execute(query, params)

    @staticmethod
    def safe_execute(func, *args, **kwargs):
        """Safely execute a function with error handling."""
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            debug_logger.exception("Safe execute failed", e)
            return False, None, str(e)
