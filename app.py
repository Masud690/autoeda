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

# ---------------- CONFIG ----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AutoEDA Pro", page_icon="📊", layout="wide")

# ---------------- UI ----------------
st.markdown("""
<style>
.stApp {background-color:#0e1117; color:white;}
h1,h2,h3 {color:#00e5ff;}
.stMetric {background:#1c1f26; padding:12px; border-radius:10px;}
.stButton>button {background:#00e5ff; color:black; font-weight:bold;}
</style>
""", unsafe_allow_html=True)

st.title("🚀 AutoEDA Pro")
st.caption("AI-powered Automated Data Analysis Platform")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Upload", "Overview", "Analysis", "AI Dashboard", "AI Report", "Export"]
)

# ---------------- UPLOAD ----------------
if section == "Upload":
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file:
        st.session_state.df = pd.read_csv(file)
        st.success("File uploaded!")
        st.dataframe(st.session_state.df.head())

# ---------------- CHECK ----------------
if "df" not in st.session_state:
    st.info("Upload dataset first")
    st.stop()

df = st.session_state.df
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

# ---------------- OVERVIEW ----------------
if section == "Overview":
    st.header("Dataset Overview")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(num_cols))
    c4.metric("Categorical", len(cat_cols))
    c5.metric("Missing %", round(df.isnull().mean().mean()*100,1))
    st.dataframe(df.head(10))

# ---------------- ANALYSIS ----------------
if section == "Analysis":
    st.header("Data Analysis")

    st.subheader("Missing Values")
    miss = df.isnull().sum()
    miss = miss[miss>0]
    if miss.empty:
        st.success("No missing values")
    else:
        st.bar_chart(miss)

    st.subheader("Statistics")
    st.dataframe(df.describe().T)

    st.subheader("Distribution")
    col = st.selectbox("Column", num_cols)
    st.plotly_chart(px.histogram(df, x=col), use_container_width=True)

    if len(num_cols)>=2:
        st.subheader("Correlation")
        st.plotly_chart(px.imshow(df[num_cols].corr(), text_auto=True))

    if cat_cols:
        st.subheader("Categorical")
        cat = st.selectbox("Category", cat_cols)
        st.plotly_chart(px.bar(df[cat].value_counts()))

# ---------------- AI DASHBOARD ----------------
if section == "AI Dashboard":
    st.header("AI Smart Dashboard")

    if not api_key:
        st.error("Missing GROQ API KEY")
    else:
        if st.button("Run AI Analysis"):
            client = Groq(api_key=api_key)

            prompt = f"Suggest 4 useful charts for dataset columns {df.columns.tolist()} in JSON"

            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}]
            )

            st.write(res.choices[0].message.content)

# ---------------- AI REPORT ----------------
if section == "AI Report":
    st.header("AI Report")

    if not api_key:
        st.error("Missing GROQ API KEY")
    else:
        if st.button("Generate Report"):
            client = Groq(api_key=api_key)

            prompt = f"Write detailed dataset analysis for columns {df.columns.tolist()}"

            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}]
            )

            st.session_state.report = res.choices[0].message.content
            st.write(st.session_state.report)

# ---------------- PDF SAFE FUNCTIONS ----------------
def clean_text(text):
    # Replace unicode
    replacements = {
        '\u2014': '-', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2022': '-',
        '\u00a0': ' ', '\u20b9': 'Rs'
    }
    for k,v in replacements.items():
        text = text.replace(k,v)

    # Remove non-latin chars
    text = text.encode('latin-1','ignore').decode('latin-1')

    # Remove super long words
    text = re.sub(r'\S{50,}', 'LONG_WORD_REMOVED', text)

    return text


def safe_write(pdf, text):
    for line in text.split("\n"):
        if not line.strip():
            pdf.ln(4)
            continue

        words = line.split(" ")
        current = ""

        for word in words:
            test = current + " " + word if current else word

            try:
                pdf.multi_cell(0, 5, test)
                current = test
            except:
                pdf.multi_cell(0, 5, current)
                current = word

        if current:
            pdf.multi_cell(0, 5, current)

# ---------------- EXPORT ----------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ---------------- EXPORT ----------------
if section == "Export":
    st.header("Download Report")

    if "report" not in st.session_state:
        st.info("Generate report first")
    else:
        if st.button("Generate PDF"):

            doc = SimpleDocTemplate("report.pdf", pagesize=letter)
            styles = getSampleStyleSheet()

            story = []

            text = st.session_state.report

            for line in text.split("\n"):
                story.append(Paragraph(line, styles["Normal"]))
                story.append(Spacer(1, 10))

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                doc.filename = tmp.name
                doc.build(story)

                with open(tmp.name, "rb") as f:
                    st.download_button(
                        "Download PDF",
                        f,
                        file_name="report.pdf",
                        mime="application/pdf"
                    )

                os.unlink(tmp.name)
