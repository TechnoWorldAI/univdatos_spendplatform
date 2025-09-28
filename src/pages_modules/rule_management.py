import streamlit as st
import pandas as pd
import sqlite3
import io
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute
from .db import DB_NAME, insert_data


def render_page():
    #st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> ⚙️ Rules Management </div>", unsafe_allow_html=True)
    # Show all rows in rule_table and make editable
    with get_db_connection() as conn:
        rule_df = pd.read_sql("SELECT * FROM rule_table", conn)
    #st.markdown("<div style='font-size:1.1em; color:#444; margin-bottom:18px;'> Existing Rules </div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    card_style = "background-color:#f8f9fa; border-radius:6px; box-shadow:0 1px 4px grey; padding:0px; margin-bottom:20px; text-align:center;"
    with col1:
        st.markdown(f"<div style='{card_style}'><span style='font-size:0.95em; color:#333;'>Total Rules</span><br><span style='font-size:1.15em; font-weight:bold; color:#007bff;'>{len(rule_df)}</span></div>", unsafe_allow_html=True)
    with col2:
        active_rules = rule_df[rule_df['active_flag'] == 1]
        st.markdown(f"<div style='{card_style}'><span style='font-size:0.95em; color:#333;'>Active Rules</span><br><span style='font-size:1.15em; font-weight:bold; color:green;'>{len(active_rules)}</span></div>", unsafe_allow_html=True)
    with col3:
        deleted_rules = rule_df[rule_df['is_deleted'] == 1]
        st.markdown(f"<div style='{card_style}'><span style='font-size:0.95em; color:#333;'>Deleted Rules</span><br><span style='font-size:1.15em; font-weight:bold; color:#dc3545;'>{len(deleted_rules)}</span></div>", unsafe_allow_html=True)
    #st.markdown("<hr style='border:0.5px solid #D8D8D8; margin-top:10px;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:1.0em; color:#444; margin-bottom:18px;'> Existing Rules </div>", unsafe_allow_html=True)
    render_styled_table(rule_df)
    # edited_df = st.data_editor(
    #     rule_df,
    #     num_rows="dynamic",       # allow adding rows
    #     width="stretch",
    #     key="rule_data_editor",
    #     column_config={           # define table design
    #         "rule_id": st.column_config.NumberColumn(
    #             "Rule ID",
    #             help="Unique identifier for each rule",
    #             disabled=True,
    #             width="small"
    #         ),
    #         "rule_description": st.column_config.TextColumn(
    #             "Description",
    #             help="What the rule does"
    #         ),
    #         "rule_data": st.column_config.TextColumn(
    #             "Data",
    #             help="Data used in the rule"
    #         ),
    #         "rule_condition": st.column_config.TextColumn(
    #             "Condition",
    #             help="Logical condition applied in rule"
    #         ),
    #         "active_flag": st.column_config.NumberColumn(
    #             "Is Active",
    #             help="Indicates if the rule is active"
    #         ),
    #         "created_on": st.column_config.DatetimeColumn(
    #             "Created On",
    #             format="YYYY-MM-DD HH:mm",
    #             help="Timestamp when the rule was created",
    #             disabled=True
    #         ),
    #         "is_deleted": st.column_config.NumberColumn(
    #             "Is Deleted",
    #             help="Indicates if the rule is deleted"
    #         ),
    #     },
    #     hide_index=True
    # )

    # if st.button("Save Changes", key="save_rule_edit_btn"):
    #     with get_db_connection() as conn:
    #         edited_df.to_sql("rule_table", conn, if_exists="replace", index=False)
    #     st.success("✅ Edits saved to rule_table!")
    #     st.rerun()  # refresh page to reload latest DB data
    
    st.markdown("<hr style='border:0.5px solid #D8D8D8; margin-top:6px;'>", unsafe_allow_html=True)
    # To add new rule
    st.markdown("<div style='font-size:1.1em; color:#444; margin-bottom:12px'> Add New Rule </div>", unsafe_allow_html=True)
    #st.info("Form is rendered. If you see this message, the Add Rule button should be visible below.")

    with st.form(key="add_rule_form"):
        rule_description = st.text_input("Rule Description", key="rule_desc_input")
        rule_data = st.text_input("Rule Data", key="rule_data_input")
        rule_condition = st.text_input("Rule Condition", key="rule_condition_input")

        submitted = st.form_submit_button("Add Rule")
        if submitted:
            try:
                with get_db_connection() as conn:
                    conn.execute(
                        """
                        INSERT INTO rule_table (rule_description, rule_data, rule_condition, active_flag, created_on, is_deleted)
                        VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, 0)
                        """,
                        (rule_description, rule_data, rule_condition)
                    )
                    conn.commit()
                st.success("✅ New rule added!")
                st.session_state.rule_description = ""
                st.session_state.rule_data = ""
                st.session_state.rule_condition = ""
                st.rerun()  # refresh page so new row shows up
            except Exception as e:
                st.error(f"Error inserting rule: {e}")


def render_styled_table(df):
    column_mapping = {
        "rule_id": "Rule ID",
        "rule_description": "Description",
        "rule_data": "Data",
        "rule_condition": "Condition",
        "active_flag": "Is Active",
        "created_on": "Created On",
        "is_deleted": "Is Deleted",
    }

    df = df.rename(columns=column_mapping)

    styled_df = (
        df.style
        .hide(axis="index")
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#1D2951"),
                ("color", "white"),
                ("text-align", "center"),
                ("border", "1px solid black"),
                ("padding", "1px"),
                ("font-size", "0.85em"),
                ("width", "14.2%")
            ]},
            {"selector": "td", "props": [
                ("border", "1px solid black"),
                ("text-align", "center"),
                ("padding", "4px"),
                ("font-size", "0.8em")
            ]},
            {"selector": "table", "props": [
                ("border-collapse", "expand"),
                ("border-radius", "6px"),
                ("width", "100%")
            ]}
        ])
    )
    html_table = styled_df.to_html(escape=False, table_attributes='class="dataframe"')

    # Wrap in scrollable container
    scrollable_html = f'<div style="max-height:300px; overflow:auto; border:0px solid #ddd; padding:5px;">{html_table}</div>'

    st.markdown(scrollable_html, unsafe_allow_html=True)
#if __name__ == "__main__":
#    render_page()