import json
import re
import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from fpdf import FPDF
from groq import Groq
from dotenv import load_dotenv

# ------------------ CONFIG ------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AutoEDA Pro", page_icon="📊", layout="wide")

# ------------------ UI STYLE ------------------
st.markdown("""
<style>
    .stApp {background-color: #0e1117; color: white;}
    h1,h2,h3 {color:#00e5ff;}
    .stMetric {background:#1c1f26; padding:15px; border-radius:10px;}
    .stButton>button {background:#00e5ff; color:black; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("🚀 AutoEDA Pro")
st.caption("AI-powered Automated Data Analysis Platform")

# ------------------ SIDEBAR ------------------
st.sidebar.title("📂 Navigation")
section = st.sidebar.radio(
    "Go to",
    ["📤 Upload", "📊 Overview", "🔍 Analysis", "🧠 AI Dashboard", "🤖 Report", "📄 Export"]
)

# ------------------ UPLOAD ------------------
if section == "📤 Upload":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        st.success("✅ File uploaded!")

        if "df" not in st.session_state:
            st.session_state.df = pd.read_csv(file)

        st.dataframe(st.session_state.df.head(), use_container_width=True)

# ------------------ CHECK ------------------
if "df" not in st.session_state:
    st.info("👈 Upload dataset first")
    st.stop()

df = st.session_state.df
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# ------------------ OVERVIEW ------------------
if section == "📊 Overview":
    st.header("📊 Dataset Overview")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(num_cols))
    c4.metric("Categorical", len(cat_cols))
    c5.metric("Missing %", round(df.isnull().mean().mean()*100,1))

    st.dataframe(df.head(10), use_container_width=True)

# ------------------ ANALYSIS ------------------
if section == "🔍 Analysis":
    st.header("🔍 Data Analysis")

    # Missing
    st.subheader("Missing Values")
    miss = df.isnull().sum()
    miss = miss[miss>0]
    if miss.empty:
        st.success("No missing values")
    else:
        st.bar_chart(miss)

    # Stats
    st.subheader("Statistics")
    st.dataframe(df.describe().T)

    # Distribution
    st.subheader("Distribution")
    col = st.selectbox("Column", num_cols)
    fig = px.histogram(df, x=col)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation
    if len(num_cols)>=2:
        st.subheader("Correlation")
        fig2 = px.imshow(df[num_cols].corr(), text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

    # Categorical
    if cat_cols:
        st.subheader("Categorical")
        cat = st.selectbox("Category", cat_cols)
        fig3 = px.bar(df[cat].value_counts())
        st.plotly_chart(fig3, use_container_width=True)

# ------------------ AI DASHBOARD ------------------
if section == "🧠 AI Dashboard":
    st.header("🧠 AI Smart Dashboard")

    if not api_key:
        st.error("Missing GROQ API KEY")
    else:
        if st.button("Run AI Analysis"):
            client = Groq(api_key=api_key)

            prompt = f"Suggest 4 charts for dataset columns {df.columns.tolist()} in JSON"

            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}]
            )

            st.write(res.choices[0].message.content)

# ------------------ REPORT ------------------
if section == "🤖 Report":
    st.header("🤖 AI Report")

    if not api_key:
        st.error("Missing GROQ API KEY")
    else:
        if st.button("Generate Report"):
            client = Groq(api_key=api_key)

            prompt = f"Write dataset analysis for columns {df.columns.tolist()}"

            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}]
            )

            st.session_state.report = res.choices[0].message.content
            st.write(st.session_state.report)

# ------------------ EXPORT ------------------
if section == "📄 Export":
    st.header("📄 Download Report")

    if "report" not in st.session_state:
        st.info("Generate report first")
    else:
        if st.button("Generate PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)

            for line in st.session_state.report.split("\n"):
                pdf.multi_cell(0, 5, line)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                pdf.output(tmp.name)

                with open(tmp.name, "rb") as f:
                    st.download_button("Download PDF", f, file_name="report.pdf")

                os.unlink(tmp.name)
