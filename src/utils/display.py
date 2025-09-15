"""Display helpers for Streamlit: normalize DataFrames before display to avoid pyarrow warnings.

Functions:
- normalize_df_for_display(df): returns a shallow copy with id columns cast to pandas nullable Int64 and timestamps formatted as ISO strings.
"""
from typing import List
import pandas as pd

ID_COLUMNS = ['category_id', 'parent_category_id', 'vendor_id', 'user_id']
TS_COLUMNS = ['created_at', 'updated_at']


def normalize_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    display_df = df.copy()
    try:
        for c in ID_COLUMNS:
            if c in display_df.columns:
                try:
                    display_df[c] = pd.to_numeric(display_df[c], errors='coerce').astype('Int64')
                except Exception:
                    # best-effort: leave as-is if conversion fails
                    pass

        for t in TS_COLUMNS:
            if t in display_df.columns:
                try:
                    # convert to string to avoid timezone/pyarrow issues
                    display_df[t] = pd.to_datetime(display_df[t], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    try:
                        display_df[t] = display_df[t].astype(str)
                    except Exception:
                        pass

    except Exception:
        # If anything goes wrong, return original df to avoid breaking UI
        return df

    return display_df
