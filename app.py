import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page config (VERY IMPORTANT for UI)
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Load model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Load dataset
df = pd.read_csv('laptop_data.csv')

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align: center;'>💻 Laptop Price Predictor</h1>
    <p style='text-align: center; font-size:18px; color:gray;'>
        Configure your laptop and get an instant price prediction
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Laptop Configuration")

company = st.sidebar.selectbox("Brand", df['Company'].unique())
type = st.sidebar.selectbox("Laptop Type", df['TypeName'].unique())

ram = st.sidebar.selectbox("RAM", [2,4,6,8,12,16,24,32,64])
weight = st.sidebar.number_input("Weight (kg)", min_value=0.5, step=0.1)

inches = st.sidebar.slider("Screen Size (Inches)", 10.0, 18.0, 13.0)

resolution = st.sidebar.selectbox(
    "Screen Resolution",
    df['ScreenResolution'].unique()
)

cpu = st.sidebar.selectbox("Processor", df['Cpu'].unique())
gpu = st.sidebar.selectbox("Graphics Card", df['Gpu'].unique())
memory = st.sidebar.selectbox("Storage", df['Memory'].unique())
os = st.sidebar.selectbox("Operating System", df['OpSys'].unique())

predict_button = st.sidebar.button("🔮 Predict Price")

# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🖥 Selected Configuration")

    st.markdown(f"**Brand:** {company}")
    st.markdown(f"**Type:** {type}")
    st.markdown(f"**RAM:** {ram} GB")
    st.markdown(f"**Weight:** {weight} kg")
    st.markdown(f"**Screen Size:** {inches} inches")
    st.markdown(f"**Resolution:** {resolution}")
    st.markdown(f"**CPU:** {cpu}")
    st.markdown(f"**GPU:** {gpu}")
    st.markdown(f"**Storage:** {memory}")
    st.markdown(f"**OS:** {os}")

with col2:
    st.subheader("💰 Price Prediction")

    if predict_button:

        with st.spinner("Analyzing configuration..."):

            query = pd.DataFrame({
                'Unnamed: 0': [0],
                'Company': [company],
                'TypeName': [type],
                'Inches': [inches],
                'ScreenResolution': [resolution],
                'Cpu': [cpu],
                'Ram': [str(ram) + "GB"],
                'Memory': [memory],
                'Gpu': [gpu],
                'OpSys': [os],
                'Weight': [str(weight) + "kg"]
            })

            prediction = pipe.predict(query)[0]

        st.success("Prediction Complete ✅")

        st.markdown(
            f"""
            <div style="
                background-color:#111;
                padding:30px;
                border-radius:15px;
                text-align:center;
            ">
                <h2 style="color:gray;">Estimated Price</h2>
                <h1 style="color:#00FFAA;">₹ {int(prediction):,}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("👈 Click **Predict Price** from the sidebar")

# ---------- FOOTER ----------
st.divider()

st.markdown(
    """
    <p style='text-align:center; color:gray;'>
        Built with Streamlit • Machine Learning Regression Model
    </p>
    """,
    unsafe_allow_html=True
)
