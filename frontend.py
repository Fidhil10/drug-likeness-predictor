import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import os

# Page config
st.set_page_config(
    page_title="DrugAI - Precision Prediction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background-color: #0e1117;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e2a4a, #0e1117);
    }
    
    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        margin-bottom: 20px;
    }
    
    .predict-box {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        font-weight: 600;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 5px solid #6366f1;
        padding: 20px;
        border-radius: 10px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for Analytics
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar Navigation & Input
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>💊 DrugAI</h1>", unsafe_allow_html=True)
    st.write("---")
    st.subheader("Patient Metrics")
    st.write("Enter clinical data for drug recommendation.")
    
    with st.form("patient_form"):
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        sex = st.selectbox("Sex", ["F", "M"])
        bp = st.selectbox("Blood Pressure", ["HIGH", "LOW", "NORMAL"])
        chol = st.selectbox("Cholesterol", ["HIGH", "NORMAL"])
        na_to_k = st.number_input("Na to K Ratio", min_value=0.0, value=15.0, format="%.3f")
        
        submit = st.form_submit_button("Generate Prediction")

# Hero Section
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("<h1 style='font-size: 3.5rem; margin-bottom: 0; line-height: 1.2;'>DrugAI Prediction <br><span style='color: #6366f1;'>Engine</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.2rem; opacity: 0.8;'>Precise pharmaceutical recommendations powered by AI.</p>", unsafe_allow_html=True)
with col2:
    # Display hero image
    hero_img_path = r"E:\PROJECTS\drug_pred\images\drug_pred.png"
    if os.path.exists(hero_img_path):
        st.image(hero_img_path, use_container_width=True)
    else:
        st.empty()

# Main Content Tabs
tab1, tab2 = st.tabs(["🎯 Prediction", "📊 Analytics"])

with tab1:
    if submit:
        # Call Backend API
        payload = {
            "Age": int(age),
            "Sex": sex,
            "BP": bp,
            "Cholesterol": chol,
            "Na_to_K": float(na_to_k)
        }
        
        try:
            with st.spinner("Analyzing patient profile..."):
                response = requests.post("http://localhost:8000/predict", json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    drug = result["prediction_drug"]
                    
                    st.markdown(f"""
                    <div class="predict-box">
                        <h3>Recommended Medication</h3>
                        <h1 style="font-size: 4rem; margin: 10px 0;">{drug}</h1>
                        <p>Inference result based on Clinical Decision Tree Model</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Log to history
                    st.session_state.history.append({
                        "Age": age,
                        "Sex": sex,
                        "BP": bp,
                        "Chol": chol,
                        "Ratio": na_to_k,
                        "Prediction": drug
                    })
                else:
                    st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")
            st.info("Ensure the backend server is running on port 8000.")
    else:
        st.info("👈 Enter patient details in the sidebar to get started.")
        
        # Display current parameters
        st.markdown("### Current Input Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.markdown(f'<div class="metric-card">Age<br><b>{age}</b></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="metric-card">Sex<br><b>{sex}</b></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="metric-card">BP<br><b>{bp}</b></div>', unsafe_allow_html=True)
        m4.markdown(f'<div class="metric-card">Cholesterol<br><b>{chol}</b></div>', unsafe_allow_html=True)

with tab2:
    st.header("Real-time Analytics")
    
    if not st.session_state.history:
        st.warning("No prediction history found. Start predicting to see insights.")
    else:
        df = pd.DataFrame(st.session_state.history)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("Drug Prediction Distribution")
            fig = px.pie(df, names="Prediction", hole=0.4, 
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font_color="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with c2:
            st.subheader("Na/K Ratio vs Age Analysis")
            fig = px.scatter(df, x="Age", y="Ratio", color="Prediction", 
                             size=[15]*len(df), hover_data=["Sex", "BP"])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Detailed History Logs")
        st.dataframe(df.style.set_properties(**{'background-color': '#1e2a4a', 'color': 'white', 'border-color': 'white'}), use_container_width=True)
        
        if st.button("Purge Session History"):
            st.session_state.history = []
            st.success("History cleared!")

st.markdown("---")
st.markdown("<p style='text-align: center; opacity: 0.5;'>Precision Pharma Dashboard | Built with streamlit and FastAPI</p>", unsafe_allow_html=True)
