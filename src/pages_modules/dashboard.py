"""Reports page module for analytics and reporting."""
import streamlit as st
import pandas as pd
import plotly.express as px
import io
from src.utils.db_simple import get_db_connection
from src.config import config
from src.utils.debug import debug_logger, show_error_block, safe_execute


def render_page() -> None:
    """Render the reports and analytics page."""
    try:
        debug_logger.info("Rendering reports page")
        #st.markdown("<div style='font-size:1.2em; color:#444; margin-bottom:18px;'> </div>", unsafe_allow_html=True)
        # Load data
        debug_logger.debug("Loading transaction data for reports")
        success, df, error = safe_execute(
            load_transaction_data,
            error_title="Failed to Load Report Data",
            show_ui_error=True
        )
        
        if not success or df is None or df.empty:
            debug_logger.warning("No data available for reporting")
            st.warning("No data available for reporting.")
            return
        
        debug_logger.debug("Report data loaded", {"rows": len(df), "columns": len(df.columns)})
        
        # Filters
        debug_logger.debug("Rendering filters")
        safe_execute(render_filters, df, error_title="Failed to Render Filters", show_ui_error=True)
        st.markdown("<hr style='border:0.5px solid #D8D8D8; margin:0px;'>", unsafe_allow_html=True)
        
        # Apply filters from session state
        debug_logger.debug("Applying filters")
        success, filtered_df, error = safe_execute(
            apply_filters, df,
            error_title="Failed to Apply Filters",
            show_ui_error=True
        )
        
        if not success or filtered_df is None:
            return
        
        debug_logger.debug("Filters applied", {"filtered_rows": len(filtered_df)})
        
        # Metric cards
        safe_execute(render_spend_summary, filtered_df, error_title="Failed to Render Spend Summary", show_ui_error=True)

        # Add spacing below cards
        st.markdown("<div style='margin-bottom:18px;'></div>", unsafe_allow_html=True)

        # Reports (charts)  
        col1, col2, col3 = st.columns(3)
        with col1:
            safe_execute(render_spend_by_region, filtered_df, error_title="Failed to Render Spend by Region", show_ui_error=True)
            safe_execute(render_spend_monthly, filtered_df, error_title="Failed to Render Spend Monthly", show_ui_error=True)
        with col2:
            safe_execute(render_top_supplier, filtered_df, error_title="Failed to Render Top Supplier", show_ui_error=True)
            safe_execute(render_spend_by_business_unit, filtered_df, error_title="Failed to Render Spend by Business Unit", show_ui_error=True)
        with col3:
            safe_execute(render_top_category, filtered_df, error_title="Failed to Render Top Category", show_ui_error=True)
            safe_execute(render_top_item, filtered_df, error_title="Failed to Render Top Item", show_ui_error=True)


        # Export options
        safe_execute(render_export_options, filtered_df, error_title="Failed to Render Export Options", show_ui_error=True)
        
    except Exception as e:
        debug_logger.exception("Error rendering reports page", e)
        show_error_block("Reports Page Error", e)

def render_filters(df: pd.DataFrame) -> None:
    """Render filter controls."""
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # Invoice Date filter (calendar format)
        min_date = pd.to_datetime(df['invoice_date']).min() if 'invoice_date' in df.columns and not df.empty else None
        max_date = pd.to_datetime(df['invoice_date']).max() if 'invoice_date' in df.columns and not df.empty else None
        selected_invoice_date = st.date_input(
            "Date",
            value=None,
            min_value=min_date,
            max_value=max_date,
            key="invoice_date_filter"
        )
    with col2:
        # Region filter
        regions = df['region'].unique().tolist() if 'region' in df.columns else []
        selected_regions = st.multiselect(
            "Region",
            options=regions,
            default=None,
            key="region_filter"
        )
    with col3:
        # Supplier filter
        suppliers = df['business_unit'].unique().tolist() if 'business_unit' in df.columns else []
        selected_suppliers = st.multiselect(
            "Business Unit",
            options=suppliers[:20],  # Limit to first 20 for performance
            default=None,
            key="business_unit_filter"
        )
    with col4:
        # Category filter
        invoice_type = df['invoice_item_type'].unique().tolist() if 'invoice_item_type' in df.columns else []
        selected_invoice_type = st.multiselect(
            "Invoice Item Type",
            options=invoice_type,
            default=None,
            key="invoice_item_type_filter"
        )
    with col5:
        # Category filter
        categories = df['category_2'].unique().tolist() if 'category_2' in df.columns else []
        selected_categories = st.multiselect(
            "Category",
            options=categories,
            default=None,
            key="category_2_filter"
        )

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filters based on session state."""
    filtered_df = df.copy()
    
    # Apply date filter
    if 'invoice_date_filter' in st.session_state and st.session_state.invoice_date_filter:
        if 'invoice_date' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['invoice_date'].isin(st.session_state.invoice_date_filter)]

    # Apply region filter
    if 'region_filter' in st.session_state and st.session_state.region_filter:
        if 'region' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['region'].isin(st.session_state.region_filter)]
    
    # Apply supplier filter
    if 'business_unit_filter' in st.session_state and st.session_state.business_unit_filter:
        if 'business_unit' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['business_unit'].isin(st.session_state.business_unit_filter)]
    
    # Apply Item type filter
    if 'invoice_item_type_filter' in st.session_state and st.session_state.invoice_item_type_filter:
        if 'invoice_item_type' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['invoice_item_type'].isin(st.session_state.invoice_item_type_filter)]
    
    # Apply category filter
    if 'category_2_filter' in st.session_state and st.session_state.category_2_filter:
        if 'category_2' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category_2'].isin(st.session_state.category_2_filter)]

    return filtered_df

def render_spend_summary(df: pd.DataFrame) -> None:
    """Render spend summary metrics."""

    col1, col2, col3, col4, col5 = st.columns(5)
    
    card_style = """
        <div style="
            background-color: #CFD8FE;
            border-radius: 8px;
            box-shadow: 1px 1px 6px rgba(0,0,0,0.06);
            padding: 4px 0 2px 0;
            text-align: center;
            margin-bottom: 0px;
            width: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        ">
            <div style="font-size:0.98em; font-weight:600; color:#1F327E; margin-bottom:4px;">{title}</div>
            <div style="font-size:1.08em; font-weight:bold; color:#1F327E;">{value}</div>
        </div>
    """

    total_spend = float(df.filter(like="total_amount").sum().sum())
    if isinstance(total_spend, pd.Series):
            total_spend = total_spend.sum()
    with col1:
        st.markdown(
            card_style.format(title="Total Spend", value=f"${total_spend:,.2f}"),
            unsafe_allow_html=True
        )

    # Transactions
    transaction_count = len(df)
    with col2:
        st.markdown(
            card_style.format(title="Transactions", value=f"{transaction_count:,}"),
            unsafe_allow_html=True
        )

    # Suppliers
    supplier_count = df['supplier_details'].nunique()
    with col3:
        st.markdown(
            card_style.format(title="Active Suppliers", value=f"{supplier_count:,}"),
            unsafe_allow_html=True
        )

     # Categories
    with col4:
        category_count = df['category_2'].nunique()
        st.markdown(
            card_style.format(title="Active Categories", value=f"{category_count:,}"),
            unsafe_allow_html=True
        )
    # Active Errors
    with col5:
        error_count = 0  # Placeholder
        st.markdown(
            card_style.format(title="Active Errors", value=f"{error_count}"),
            unsafe_allow_html=True
        )

def render_spend_by_region(df: pd.DataFrame) -> None:
    """Render spend by region pie chart."""
    
    if 'region' in df.columns:
        region_spend = df.groupby('region')['total_amount'].sum().reset_index()
        fig = px.pie(
            region_spend,
            values='total_amount',
            names='region',
            title="Spend by Region",
            color_discrete_sequence=['#355C7D']
        )
        # Move legend to bottom
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
                font=dict(size=10)   # smaller legend font
            ),
            title=dict(
                text="Spend by Region",
                x=0.5,                # center align title horizontally
                xanchor='center',
                font=dict(size=14)
            ),
            margin=dict(l=0, r=0, t=80, b=40),  # cut margins
            height=280,   # smaller chart height
        )
        st.plotly_chart(fig, use_container_width=True)

def render_top_supplier(df: pd.DataFrame) -> None:
    """Render top suppliers bar chart."""
    top_suppliers = df.groupby('supplier_details')['total_amount'].sum().nlargest(5).reset_index()
    fig = px.bar(
        top_suppliers,
        x='total_amount',
        y='supplier_details',
        orientation='h',
        title="Top 5 Suppliers by Spend",
        color_discrete_sequence=['#003F5C']
    )
    max_value = top_suppliers['total_amount'].max()
    fig.update_layout(
        xaxis=dict(
            range=[0, max_value * 1.1],
            title_text="",
        ),
        yaxis=dict(
            categoryorder='total ascending',
            title_text="",
            ticklabelposition="outside left",
            automargin=True
        ),  # biggest at top
        title=dict(
            x=0.5,                # center align title horizontally
            xanchor='center',
            font=dict(size=14)
        ),
        margin=dict(l=80, r=20, t=80, b=40),  # cut margins
        height=280,   # smaller chart height
        #plot_bgcolor="#D9D9D9",
        #paper_bgcolor="#002366"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_top_category(df: pd.DataFrame) -> None:
    """Render top suppliers bar chart."""
    top_categories = df.groupby('category_2')['total_amount'].sum().nlargest(5).reset_index()
    fig = px.pie(
        top_categories,
        values='total_amount',
        names='category_2',
        title="Top 5 Categories by Spend",
        color_discrete_sequence=['#2F8BCC'],
        hole=0.5
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        margin=dict(l=0, r=0, t=80, b=40),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)

def render_spend_monthly(df: pd.DataFrame) -> None:
    """Render spend trend over time."""
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    monthly_spend = df.groupby(df['invoice_date'].dt.to_period('M'))['total_amount'].sum().reset_index()
    # Convert 'invoice_date' to mmm-yyyy format
    monthly_spend['invoice_date'] = monthly_spend['invoice_date'].dt.strftime('%b-%Y')
    # Convert total_amount to millions
    monthly_spend['total_amount'] = (monthly_spend['total_amount'] / 1_000_000).round(0).astype(int)

    max_value = monthly_spend['total_amount'].max()
    fig = px.line(
        monthly_spend,
        x='invoice_date',
        y='total_amount',
        title="Monthly Spend Trend",
        markers=True,
        color_discrete_sequence=['#868070']
    )
    fig.update_traces(
        line=dict(dash='solid', width=3),   # solid thick line
        marker=dict(size=6, symbol='circle')  # clean circle markers
    )
    fig.update_layout(
        xaxis=dict(
            title=""
        ),
        yaxis=dict(
            range=[0, max_value * 1.5],
            title="(Millions)",
            tickformat=",.2f"
        ),
        title=dict(
            x=0.5,                # center align title horizontally
            xanchor='center',
            font=dict(size=14)
        ),
        margin=dict(l=80, r=20, t=80, b=40),  # cut margins
        height=280,   # smaller chart height
    )
    st.plotly_chart(fig, use_container_width=True)

def render_spend_by_business_unit(df: pd.DataFrame) -> None:
    """Render top suppliers bar chart."""
    spend_bu = df.groupby('business_unit')['total_amount'].sum().nlargest(5).reset_index()
    fig = px.bar(
        spend_bu,
        x='total_amount',
        y='business_unit',
        orientation='h',
        title="Top 5 Business Units by Spend",
        color_discrete_sequence=['#717560']
    )
    max_value = spend_bu['total_amount'].max()
    fig.update_layout(
        xaxis=dict(
            range=[0, max_value * 1.1],
            title_text="",
        ),
        yaxis=dict(
            categoryorder='total ascending',
            title_text="",
            ticklabelposition="outside left",
            automargin=True
        ),  # biggest at top
        title=dict(
            x=0.5,                # center align title horizontally
            xanchor='center',
            font=dict(size=14)
        ),
        margin=dict(l=80, r=20, t=80, b=40),  # cut margins
        height=280,   # smaller chart height
        #plot_bgcolor="#D9D9D9",
        #paper_bgcolor="#002366"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_top_item(df: pd.DataFrame) -> None:
    """Render top suppliers bar chart."""
    item_totalamount = df.groupby('invoice_item_type')['total_amount'].sum().reset_index()
    fig = px.bar(
        item_totalamount,
        x='total_amount',
        y='invoice_item_type',
        orientation='h',
        title="Invoice type by Spend",
        color_discrete_sequence=['#9f4f36']
    )
    max_value = item_totalamount['total_amount'].max()
    fig.update_layout(
        xaxis=dict(
            range=[0, max_value * 1.1],
            title_text="",
        ),
        yaxis=dict(
            categoryorder='total ascending',
            title_text="",
            ticklabelposition="outside left",
            automargin=True
        ),  # biggest at top
        title=dict(
            x=0.5,                # center align title horizontally
            xanchor='center',
            font=dict(size=14)
        ),
        margin=dict(l=80, r=20, t=80, b=40),  # cut margins
        height=280,   # smaller chart height
        #plot_bgcolor="#D9D9D9",
        #paper_bgcolor="#002366"
    )
    st.plotly_chart(fig, use_container_width=True)

def render_export_options(df: pd.DataFrame) -> None:
    """Render data export options."""
    # st.markdown("<div style='font-size:1.02em; color:#444; margin-bottom:18px;'> Export Options </div>", unsafe_allow_html=True)

    col1, col2, col3,col4, col5, col6 = st.columns(6)
    with col1:
        if st.button("Export to CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"spend_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    with col2:
        if st.button("Export to Excel"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Spend Data', index=False)
            st.download_button(
                label="Download Excel",
                data=output.getvalue(),
                file_name=f"spend_report_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def load_transaction_data() -> pd.DataFrame:
    """Load transaction data for reporting."""
    with get_db_connection() as conn:
        query = """
            SELECT sd.*, cd.category_1, cd.category_2, cd.category_3, cd.category_4, cd.category_5
            FROM spend_data as sd
            LEFT JOIN categorized_data as cd
            ON sd.item_description = cd.item_description
        """
        return pd.read_sql_query(query, conn)
