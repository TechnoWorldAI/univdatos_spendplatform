import os
import streamlit as st

def render_page():
    # --- Custom CSS for buttons ---
    st.markdown("""
        <style>
        div.stButton > button {
            width: 160px; ! important;
            background-color: #1D2951; /* Default background (blue) */
            color: white;
            font-weight: 400;
            border: none;
            border-radius: 8px;
            padding: 5px 5px;
            box-shadow: 2px 4px 8px rgba(0,0,0,0.3);
            transition: all 0.3s ease-in-out;
            font-size: 0.60em;
            margin-top: -12px;
        }
        div.stButton > button:hover {
            background-color: orange; /* Hover background */
            box-shadow: none !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # --- Dashboard image paths ---
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    context_dir = os.path.join(base_dir, "context")

    dashboard_images = [
        (os.path.join(context_dir, "1_Spend Overview.jpg"), "Spend Overview"),
        (os.path.join(context_dir, "2_Spend Category.jpg"), "Spend Category"),
        (os.path.join(context_dir, "3_On-Off Contract.jpg"), "On-Off Contract"),
        (os.path.join(context_dir, "4_Contract Coverage.jpg"), "Contract Coverage"),
        (os.path.join(context_dir, "5_Contract Compliance.jpg"), "Contract Compliance"),
    ]

    # --- Button Layout ---
    col1, col2, col3, col4, col5 = st.columns(5)
    selected_idx = 0
    with col1:
        st.markdown('<div class="dashboard-btn">', unsafe_allow_html=True)
        if st.button("Spend Overview", key="dashboard_btn_0"):
            selected_idx = 0
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="dashboard-btn">', unsafe_allow_html=True)
        if st.button("Spend Category", key="dashboard_btn_1"):
            selected_idx = 1
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="dashboard-btn">', unsafe_allow_html=True)
        if st.button("On-Off Contract", key="dashboard_btn_2"):
            selected_idx = 2
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="dashboard-btn">', unsafe_allow_html=True)
        if st.button("Contract Coverage", key="dashboard_btn_3"):
            selected_idx = 3
        st.markdown('</div>', unsafe_allow_html=True)
    with col5:
        st.markdown('<div class="dashboard-btn">', unsafe_allow_html=True)
        if st.button("Contract Compliance", key="dashboard_btn_4"):
            selected_idx = 4
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Display Selected Dashboard ---
    img_path, caption = dashboard_images[selected_idx]
    if os.path.exists(img_path):
        st.image(img_path, width=850, caption=caption)
    else:
        st.warning(f"Dashboard not found: {caption}")
