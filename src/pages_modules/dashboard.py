import os
import base64
import streamlit as st

def render_page():
    # Display image from context folder
    dashboard1_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'context',
        '1_Spend Overview.jpg'
    )
    dashboard2_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'context',
        '2_Spend Category.jpg'
    )
    dashboard3_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'context',
        '3_On-Off Contract.jpg'
    )
    dashboard4_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'context',
        '4_Contract Coverage.jpg'
    )
    dashboard5_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'context',
        '5_Contract Compliance.jpg'
    )
    dashboard_images = [
        (dashboard1_path, "Spend Overview"),
        (dashboard2_path, "Spend Category"),
        (dashboard3_path, "On-Off Contract"),
        (dashboard4_path, "Contract Coverage"),
        (dashboard5_path, "Contract Compliance")
    ]
    for img_path, caption in dashboard_images:
        if os.path.exists(img_path):
            st.markdown(f"<div style='font-size:1.2em; color:#1D2951; margin-bottom:10px;'> {caption} </div>", unsafe_allow_html=True)
            st.image(img_path, width=850)
            st.markdown("<hr style='border:0.5px solid #D8D8D8; margin:0px;'>", unsafe_allow_html=True)
        else:
            st.warning(f"Dashboard image not found: {caption}")

