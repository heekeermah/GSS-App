import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import openai

# ------------------- GSS Calculation Function -------------------
def calculate_gss(df):
    df = df.copy()
    df.fillna(0, inplace=True)

    # Calculate shrub percent if not provided
    if 'Shrub %' not in df.columns and 'shrub' in df.columns and 'Total Cover' in df.columns:
        df['shrub_percent'] = (df['shrub'] / df['Total Cover']) * 100

    # Normalize selected features
    features = ['grazing_pressure', 'Shrub %', 'perennial_grass', 'available_biomass', 'bare_ground']
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
        min_val = df[feat].min()
        max_val = df[feat].max()
        df[f'{feat}_norm'] = (df[feat] - min_val) / (max_val - min_val + 1e-6)

    # Assign weights (can be adjusted)
    weights = {
        'grazing_pressure_norm': 0.3,
        'Shrub %_norm': 0.2,
        'perennial_grass_norm': 0.2,
        'available_biomass_norm': 0.2,
        'bare_ground_norm': 0.1
    }

    # Compute GSS
    df['GSS'] = 100 * (
        weights['perennial_grass_norm'] * df['perennial_grass_norm'] +
        weights['available_biomass_norm'] * df['available_biomass_norm'] +
        weights['shrub_percent_norm'] * (1 - df['shrub_percent_norm']) +
        weights['grazing_pressure_norm'] * (1 - df['grazing_pressure_norm']) +
        weights['bare_ground_norm'] * (1 - df['bare_ground_norm'])
    )
    return df

# ------------------- AI Assistant -------------------
def ai_assistant_gpt(row, api_key):
    api_key = st.secrets["openai"]["api_key"]
    openai.api_key = api_key

    prompt = (
        f"Plot: {row['Plot Name']}, GSS: {row['GSS']:.2f}, Grazing Pressure: {row['grazing_pressure']}, "
        f"Shrub %: {row['shrub_percent']:.2f}, Perennial Grass: {row['perennial_grass']}, "
        f"Available Biomass: {row['available_biomass']}, Bare Ground: {row['bare_ground']}. "
        f"Suggest rangeland management actions to improve or maintain grazing suitability."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert rangeland management advisor."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"AI error: {str(e)}"


# ------------------- File Uploader -------------------
def load_data(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df

# ------------------- Download Helper -------------------
def get_table_download_link(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="gss_results.csv">Download CSV File</a>'
    return href

# ------------------- Streamlit Web App -------------------
st.set_page_config(page_title="Grazing Suitability Score Dashboard", layout="wide")
st.title("\ud83c\udf3e Grazing Suitability Score (GSS) Calculator")

st.sidebar.header("Upload Your Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.success("File uploaded successfully!")

    with st.spinner("Calculating GSS and generating AI suggestions..."):
        df = calculate_gss(df)

        if api_key:
            df['AI_Advice'] = df.apply(lambda row: ai_assistant_gpt(row, api_key), axis=1)
        else:
            df['AI_Advice'] = "No API key provided. AI suggestions not available."

    # Dashboard Layout
    st.subheader("Sample of Processed Data")
    st.dataframe(df.head(20))

    # Visualizations
    st.subheader("GSS Distribution")
    fig = px.histogram(df, x='GSS', nbins=20, title="Distribution of Grazing Suitability Scores")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top and Bottom Plots by GSS")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 5 Plots")
        st.dataframe(df.sort_values('GSS', ascending=False).head(5))
    with col2:
        st.markdown("#### Bottom 5 Plots")
        st.dataframe(df.sort_values('GSS', ascending=True).head(5))

    # AI Feedback
    st.subheader("AI Assistance Suggestions")
    if api_key:
        for i, row in df.sort_values('GSS', ascending=True).head(5).iterrows():
            st.markdown(f"**{row['Plot Name']}**: {row['AI_Advice']}")
    else:
        st.info("Enter an OpenAI API key in the sidebar to see AI suggestions.")

    # Download
    st.subheader("Download Processed Data")
    st.markdown(get_table_download_link(df), unsafe_allow_html=True)

else:
    st.info("Awaiting file upload. Please upload a CSV or Excel file to begin.")
    st.markdown("Required columns: `Plot Name`, `grazing_pressure`, `shrub`, `Total Cover`, `perennial_grass`, `available_biomass`, `bare_ground`.")
