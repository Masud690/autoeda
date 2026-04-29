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

# Load API key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="AutoEDA", layout="wide")
st.title("⚡ autoEDA - Phase 1, 2 & Smart Dashboard")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    if "df_main" not in st.session_state or st.session_state.get("filename") != file.name:
        st.session_state["df_main"] = pd.read_csv(file)
        st.session_state["filename"] = file.name
        st.session_state.pop("charts", None)
        st.session_state.pop("narrative", None)

    df = st.session_state["df_main"]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # ── PHASE 1: Overview ─────────────────────────────
    st.subheader("📊 Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(num_cols))
    c4.metric("Categorical", len(cat_cols))
    c5.metric("Missing %", f"{round(df.isnull().mean().mean()*100, 1)}%")

    st.dataframe(df.head(10), use_container_width=True)

    # ── Missing Values ─────────────────────────────────
    st.subheader("🔍 Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("✅ No missing values found!")
    else:
        st.bar_chart(missing)

    # ── Descriptive Stats ──────────────────────────────
    st.subheader("📈 Descriptive Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

    # ── Distribution Plot ──────────────────────────────
    st.subheader("📉 Column Distribution")
    col = st.selectbox("Select numeric column", num_cols)
    fig = px.histogram(df, x=col, nbins=30, marginal="box",
                       color_discrete_sequence=["#00e5ff"])
    st.plotly_chart(fig, use_container_width=True)

    # ── Outlier Detection ──────────────────────────────
    st.subheader("🚨 Outlier Detection (IQR Method)")
    outlier_cols = []
    for c in num_cols:
        q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[c] < q1 - 1.5*iqr) | (df[c] > q3 + 1.5*iqr)]
        if len(outliers) > 0:
            st.warning(f"⚠️ **{c}** — {len(outliers)} outliers detected")
            outlier_cols.append(c)
        else:
            st.success(f"✅ **{c}** — No outliers")

    # ── Outlier Treatment ─────────────────────────────
    st.divider()
    st.subheader("🛠️ Outlier Treatment")

    if not outlier_cols:
        st.success("✅ No outliers to treat!")
    else:
        st.info(f"Columns with outliers: **{', '.join(outlier_cols)}**")

        treatment_col = st.selectbox(
            "Select column to treat",
            outlier_cols,
            key="treatment_col"
        )

        method = st.radio(
            "Choose treatment method",
            ["Cap (Winsorize)", "Remove Outlier Rows", "Keep As Is"],
            horizontal=True
        )

        if method == "Cap (Winsorize)":
            st.caption("📌 Replaces extreme values with Q1 - 1.5xIQR (lower) "
                       "and Q3 + 1.5xIQR (upper). Keeps all rows.")
        elif method == "Remove Outlier Rows":
            st.caption("📌 Deletes rows where the value is outside IQR bounds. "
                       "Reduces dataset size.")
        else:
            st.caption("📌 No changes made. Outliers kept and documented.")

        if st.button("Apply Treatment", type="primary"):
            q1 = df[treatment_col].quantile(0.25)
            q3 = df[treatment_col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df_treated = df.copy()

            if method == "Cap (Winsorize)":
                df_treated[treatment_col] = df_treated[treatment_col].clip(
                    lower=lower, upper=upper
                )
                st.success(f"✅ Capped **{treatment_col}** between "
                           f"{lower:.2f} and {upper:.2f}")
            elif method == "Remove Outlier Rows":
                before = len(df_treated)
                df_treated = df_treated[
                    (df_treated[treatment_col] >= lower) &
                    (df_treated[treatment_col] <= upper)
                ]
                after = len(df_treated)
                st.success(f"✅ Removed **{before - after} rows** "
                           f"from **{treatment_col}**. "
                           f"Dataset: {before} -> {after} rows")
            else:
                st.info(f"✅ Keeping **{treatment_col}** outliers as is.")
                df_treated = df.copy()

            st.markdown("#### 📊 Before vs After")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Treatment**")
                fig_before = px.box(df, y=treatment_col,
                                    color_discrete_sequence=["#ff6b6b"])
                st.plotly_chart(fig_before, use_container_width=True)
            with col2:
                st.write("**After Treatment**")
                fig_after = px.box(df_treated, y=treatment_col,
                                   color_discrete_sequence=["#6bcb77"])
                st.plotly_chart(fig_after, use_container_width=True)

            st.markdown("#### 📈 Stats Comparison")
            stats_df = pd.DataFrame({
                "Metric": ["Mean", "Std Dev", "Min", "Max", "Outlier Count"],
                "Before": [
                    round(df[treatment_col].mean(), 2),
                    round(df[treatment_col].std(), 2),
                    round(df[treatment_col].min(), 2),
                    round(df[treatment_col].max(), 2),
                    len(df[(df[treatment_col] < lower) |
                           (df[treatment_col] > upper)])
                ],
                "After": [
                    round(df_treated[treatment_col].mean(), 2),
                    round(df_treated[treatment_col].std(), 2),
                    round(df_treated[treatment_col].min(), 2),
                    round(df_treated[treatment_col].max(), 2),
                    len(df_treated[(df_treated[treatment_col] < lower) |
                                   (df_treated[treatment_col] > upper)])
                ]
            })
            st.dataframe(stats_df, use_container_width=True)

            st.session_state["df_main"] = df_treated
            st.session_state["df_treated"] = df_treated

            csv = df_treated.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Treated Dataset as CSV",
                data=csv,
                file_name=f"treated_{file.name}",
                mime="text/csv"
            )
            st.info("💡 Tip: You can treat multiple columns one by one.")

    # ── Correlation Heatmap ────────────────────────────
    if len(num_cols) >= 2:
        st.subheader("🔗 Correlation Heatmap")
        corr = df[num_cols].corr()
        fig2 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Categorical Counts ─────────────────────────────
    if cat_cols:
        st.subheader("🏷️ Categorical Column Counts")
        cat = st.selectbox("Select categorical column", cat_cols)
        vc = df[cat].value_counts().reset_index()
        vc.columns = ["category", "count"]
        fig3 = px.bar(vc, x="category", y="count",
                      color_discrete_sequence=["#c77dff"])
        st.plotly_chart(fig3, use_container_width=True)

    # ── PHASE 2.5: AI Smart Dashboard ─────────────────
    st.divider()
    st.subheader("🧠 AI-Driven Smart Dashboard")
    st.caption("LLM studies your columns and decides the best analysis automatically.")

    if not api_key:
        st.error("❌ GROQ_API_KEY not found in .env file.")
    else:
        if st.button("🔍 Auto-Analyse Dataset", type="primary"):
            with st.spinner("LLM is studying your dataset and deciding analysis..."):

                analysis_prompt = f"""
                You are a data analyst. You are given a dataset with the following structure:

                Column Names: {df.columns.tolist()}
                Data Types: {df.dtypes.apply(str).to_dict()}
                Sample Data (first 5 rows):
                {df.head(5).to_string()}
                Descriptive Stats:
                {df.describe().to_string()}
                Unique value counts per categorical column:
                { {c: df[c].nunique() for c in cat_cols} }

                Based on this, decide the BEST analyses to perform.
                Return ONLY a valid JSON array. No explanation, no markdown, no extra text.

                Each item must have:
                - "title": descriptive chart title
                - "type": one of ["bar", "line", "scatter", "box", "histogram", "pie"]
                - "x": column name for x axis (must exist in dataset)
                - "y": column name for y axis (must exist in dataset, use same as x for histogram)
                - "agg": one of ["mean", "sum", "count", "median", "none"]
                - "insight": one sentence business insight this chart reveals

                Strict Rules:
                - ONLY use column names from: {df.columns.tolist()}
                - bar: x=categorical, y=numeric, agg=mean/sum/count
                - scatter: x=numeric, y=numeric, agg=none
                - box: x=categorical, y=numeric, agg=none
                - histogram: x=numeric, y=same as x, agg=none
                - pie: x=categorical, y=numeric, agg=sum/count
                - line: x=date or ordered column, y=numeric, agg=mean
                - For categorical columns with more than 20 unique values, avoid pie charts
                - Return minimum 4 and maximum 6 chart suggestions
                - Think like a business analyst finding real actionable insights

                Return ONLY the JSON array, nothing else.
                """

                try:
                    client = Groq(api_key=api_key)
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a data analyst. Always respond with "
                                           "valid JSON only. No markdown, no backticks, "
                                           "no explanation. Just raw JSON array."
                            },
                            {
                                "role": "user",
                                "content": analysis_prompt
                            }
                        ],
                        temperature=0.2,
                        max_tokens=1500
                    )

                    raw = response.choices[0].message.content.strip()
                    raw = raw.replace("```json", "").replace("```", "").strip()
                    charts = json.loads(raw)
                    st.session_state["charts"] = charts
                    st.success(f"✅ LLM suggested {len(charts)} analyses!")

                except json.JSONDecodeError:
                    st.error("❌ LLM returned invalid JSON. Try clicking again.")
                    st.code(raw)
                except Exception as e:
                    st.error(f"❌ Error: {e}")

        if "charts" in st.session_state:
            charts = st.session_state["charts"]
            st.markdown("---")

            for i, chart in enumerate(charts):
                try:
                    title   = chart.get("title", f"Chart {i+1}")
                    ctype   = chart.get("type", "bar")
                    x_col   = chart.get("x")
                    y_col   = chart.get("y")
                    agg     = chart.get("agg", "mean")
                    insight = chart.get("insight", "")

                    if x_col not in df.columns:
                        st.warning(f"Skipping '{title}' — column '{x_col}' not found.")
                        continue
                    if ctype not in ["histogram"] and y_col not in df.columns:
                        st.warning(f"Skipping '{title}' — column '{y_col}' not found.")
                        continue

                    st.markdown(f"#### 📊 {title}")
                    st.caption(f"💡 **Insight:** {insight}")

                    if agg != "none" and ctype in ["bar", "line", "pie"]:
                        if agg == "mean":
                            plot_df = df.groupby(x_col)[y_col].mean().reset_index()
                        elif agg == "sum":
                            plot_df = df.groupby(x_col)[y_col].sum().reset_index()
                        elif agg == "count":
                            plot_df = df.groupby(x_col)[y_col].count().reset_index()
                        elif agg == "median":
                            plot_df = df.groupby(x_col)[y_col].median().reset_index()
                        plot_df.columns = [x_col, y_col]
                        plot_df = plot_df.sort_values(y_col, ascending=False)
                    else:
                        plot_df = df.copy()

                    if ctype == "bar":
                        fig = px.bar(plot_df, x=x_col, y=y_col,
                                     color=y_col,
                                     color_continuous_scale="Blues",
                                     title=title)
                    elif ctype == "line":
                        fig = px.line(plot_df, x=x_col, y=y_col,
                                      markers=True, title=title,
                                      color_discrete_sequence=["#00e5ff"])
                    elif ctype == "scatter":
                        try:
                            fig = px.scatter(plot_df, x=x_col, y=y_col,
                                             trendline="ols",
                                             color_discrete_sequence=["#00e5ff"],
                                             title=title, opacity=0.6)
                        except Exception:
                            fig = px.scatter(plot_df, x=x_col, y=y_col,
                                             color_discrete_sequence=["#00e5ff"],
                                             title=title, opacity=0.6)
                    elif ctype == "box":
                        if df[x_col].nunique() > 15:
                            top_cats = df[x_col].value_counts().head(15).index
                            plot_df = df[df[x_col].isin(top_cats)]
                        else:
                            plot_df = df.copy()
                        fig = px.box(plot_df, x=x_col, y=y_col,
                                     color=x_col, title=title)
                    elif ctype == "histogram":
                        fig = px.histogram(df, x=x_col, nbins=30,
                                           marginal="box",
                                           color_discrete_sequence=["#c77dff"],
                                           title=title)
                    elif ctype == "pie":
                        plot_df = plot_df.head(10)
                        fig = px.pie(plot_df, names=x_col,
                                     values=y_col, title=title)
                    else:
                        continue

                    fig.update_layout(
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#ffffff"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.divider()

                except Exception as e:
                    st.warning(f"Could not render **{title}**: {e}")
                    continue

    # ── PHASE 2: LLM Narrative ─────────────────────────
    st.divider()
    st.subheader("🤖 Phase 2 — AI Analyst Narrative")

    if not api_key:
        st.error("❌ GROQ_API_KEY not found in .env file.")
    else:
        st.success("✅ Groq API Key loaded from .env")

        if st.button("🧠 Generate AI Insights", type="primary"):
            with st.spinner("LLaMA 3 is analysing your dataset..."):

                outlier_summary = {}
                for c in num_cols:
                    q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
                    iqr = q3 - q1
                    n_out = len(df[(df[c] < q1 - 1.5*iqr) |
                                   (df[c] > q3 + 1.5*iqr)])
                    outlier_summary[c] = n_out

                corr_str = (df[num_cols].corr().to_string()
                            if len(num_cols) >= 2 else "N/A")

                stats_summary = f"""
                Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns
                Numeric Columns: {num_cols}
                Categorical Columns: {cat_cols}
                Missing Values: {df.isnull().sum().to_dict()}
                Duplicate Rows: {df.duplicated().sum()}
                Outliers per column: {outlier_summary}
                Descriptive Stats:
                {df.describe().to_string()}
                Correlation Matrix:
                {corr_str}
                Sample Data (first 3 rows):
                {df.head(3).to_string()}
                """

                prompt = f"""
                You are a senior data analyst writing a professional EDA report.
                Based on the dataset statistics below, write a detailed narrative report
                with the following sections:

                1. Dataset Overview
                2. Data Quality Assessment
                3. Key Patterns and Insights
                4. Correlation Analysis
                5. Outlier Analysis
                6. Business Recommendations
                7. Suggested Next Steps

                Be specific, professional, and reference actual column names and numbers.
                Do not use markdown symbols like **, ##, or bullet symbols.
                Write in plain text only.

                DATASET STATISTICS:
                {stats_summary}
                """

                try:
                    client = Groq(api_key=api_key)
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a senior data analyst. "
                                           "Write in plain text only. "
                                           "No markdown, no special symbols, no bullet points."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.4,
                        max_tokens=2000
                    )

                    narrative = response.choices[0].message.content

                    st.markdown("---")
                    st.markdown("### 📝 AI Generated Analysis Report")
                    st.markdown(narrative)

                    usage = response.usage
                    st.caption(f"Tokens used — Prompt: {usage.prompt_tokens} | "
                               f"Completion: {usage.completion_tokens} | "
                               f"Total: {usage.total_tokens}")

                    st.session_state["narrative"] = narrative
                    st.session_state["df_shape"] = df.shape
                    st.session_state["filename"] = file.name

                    st.success("✅ Report generated! Generate PDF below.")

                except Exception as e:
                    st.error(f"❌ Groq API Error: {e}")

    # ── PHASE 3: PDF Report ────────────────────────────
    st.divider()
    st.subheader("📄 Phase 3 — Download PDF Report")

    if "narrative" not in st.session_state:
        st.info("💡 Generate the AI Narrative first (Phase 2) to enable PDF download.")
    else:
        if st.button("📥 Generate & Download PDF Report", type="primary"):
            with st.spinner("Building your PDF report..."):

                def clean_text(text):
                    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
                    text = re.sub(r'\*(.*?)\*', r'\1', text)
                    text = re.sub(r'#{1,6}\s*', '', text)
                    replacements = {
                        '\u2014': '-',
                        '\u2013': '-',
                        '\u2018': "'",
                        '\u2019': "'",
                        '\u201c': '"',
                        '\u201d': '"',
                        '\u2022': '-',
                        '\u2026': '...',
                        '\u00b0': ' degrees',
                        '\u00e9': 'e',
                        '\u00e8': 'e',
                        '\u00ea': 'e',
                        '\u00e0': 'a',
                        '\u00e2': 'a',
                        '\u00f9': 'u',
                        '\u00fb': 'u',
                        '\u20b9': 'Rs',
                        '\u00a0': ' ',
                        '\u2192': '->',
                        '\u2190': '<-',
                        '\u2191': '^',
                        '\u2193': 'v',
                        '\u00d7': 'x',
                        '\u00f7': '/',
                    }
                    for unicode_char, replacement in replacements.items():
                        text = text.replace(unicode_char, replacement)
                    text = text.encode('latin-1', 'replace').decode('latin-1')
                    return text

                class EDA_PDF(FPDF):
                    def header(self):
                        self.set_fill_color(15, 15, 25)
                        self.rect(0, 0, 210, 30, 'F')
                        self.set_font("Helvetica", "B", 16)
                        self.set_text_color(0, 100, 200)
                        self.set_xy(10, 8)
                        self.cell(0, 10,
                                  "AutoEDA - AI Generated Analysis Report",
                                  ln=True)
                        self.set_font("Helvetica", "", 9)
                        self.set_text_color(150, 150, 150)
                        self.set_x(10)
                        shape = st.session_state.get("df_shape", ("?", "?"))
                        fname = st.session_state.get("filename", "Unknown")
                        self.cell(0, 6,
                                  f"Dataset: {fname}   |   "
                                  f"Rows: {shape[0]}   |   "
                                  f"Columns: {shape[1]}",
                                  ln=True)
                        self.ln(5)

                    def footer(self):
                        self.set_y(-15)
                        self.set_font("Helvetica", "I", 8)
                        self.set_text_color(150, 150, 150)
                        self.cell(0, 10,
                                  f"Generated by AutoEDA  |  Page {self.page_no()}",
                                  align="C")

                pdf = EDA_PDF()
                pdf.set_auto_page_break(auto=True, margin=20)
                pdf.add_page()

                filename = st.session_state.get("filename", "Unknown")
                shape    = st.session_state.get("df_shape", ("?", "?"))

                # ── Section 1: Dataset Summary ─────────
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(0, 100, 200)
                pdf.cell(0, 10, "1. Dataset Summary", ln=True)
                pdf.set_draw_color(0, 100, 200)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(3)

                summary_data = [
                    ("File Name",        filename),
                    ("Total Rows",       str(shape[0])),
                    ("Total Columns",    str(shape[1])),
                    ("Numeric Cols",     str(len(num_cols))),
                    ("Categorical Cols", str(len(cat_cols))),
                    ("Missing Values",   "None detected"),
                    ("Duplicate Rows",   str(df.duplicated().sum())),
                ]

                for label, value in summary_data:
                    pdf.set_text_color(80, 80, 80)
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(60, 8, label + ":", ln=False)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.set_text_color(30, 30, 30)
                    pdf.cell(0, 8, value, ln=True)

                pdf.ln(5)

                # ── Section 2: Descriptive Statistics ──
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(0, 100, 200)
                pdf.cell(0, 10, "2. Descriptive Statistics", ln=True)
                pdf.set_draw_color(0, 100, 200)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(3)

                desc = df[num_cols].describe().round(2)
                col_w  = 32
                stat_w = 24

                pdf.set_fill_color(0, 100, 200)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(col_w, 8, "Column", border=1, fill=True)
                for stat in ["Mean", "Std", "Min", "25%", "50%", "75%", "Max"]:
                    pdf.cell(stat_w, 8, stat, border=1,
                             fill=True, align="C")
                pdf.ln()

                for col_name in num_cols:
                    pdf.set_fill_color(240, 245, 255)
                    pdf.set_text_color(30, 30, 30)
                    pdf.set_font("Helvetica", "B", 8)
                    pdf.cell(col_w, 7, col_name[:16], border=1, fill=True)
                    pdf.set_font("Helvetica", "", 8)
                    for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                        val = (str(desc.loc[stat, col_name])
                               if col_name in desc.columns else "-")
                        pdf.cell(stat_w, 7, val, border=1, align="C")
                    pdf.ln()

                pdf.ln(5)

                # ── Section 3: Outlier Summary ──────────
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(0, 100, 200)
                pdf.cell(0, 10, "3. Outlier Summary (IQR Method)", ln=True)
                pdf.set_draw_color(0, 100, 200)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(3)

                pdf.set_fill_color(0, 100, 200)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(70, 8, "Column",        border=1, fill=True)
                pdf.cell(50, 8, "Outlier Count", border=1, fill=True, align="C")
                pdf.cell(70, 8, "Status",        border=1, fill=True, align="C")
                pdf.ln()

                for c in num_cols:
                    q1    = df[c].quantile(0.25)
                    q3    = df[c].quantile(0.75)
                    iqr   = q3 - q1
                    n_out = len(df[
                        (df[c] < q1 - 1.5*iqr) |
                        (df[c] > q3 + 1.5*iqr)
                    ])
                    status = "Outliers Found" if n_out > 0 else "Clean"
                    if n_out > 0:
                        pdf.set_fill_color(255, 235, 235)
                    else:
                        pdf.set_fill_color(240, 255, 240)
                    pdf.set_text_color(30, 30, 30)
                    pdf.set_font("Helvetica", "", 9)
                    pdf.cell(70, 7, c,          border=1, fill=True)
                    pdf.cell(50, 7, str(n_out), border=1, fill=True, align="C")
                    pdf.cell(70, 7, status,     border=1, fill=True, align="C")
                    pdf.ln()

                pdf.ln(5)

                # ── Section 4: Top Correlations ─────────
                if len(num_cols) >= 2:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.set_text_color(0, 100, 200)
                    pdf.cell(0, 10, "4. Top Correlations", ln=True)
                    pdf.set_draw_color(0, 100, 200)
                    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                    pdf.ln(3)

                    corr_matrix = df[num_cols].corr().abs()
                    corr_pairs  = (
                        corr_matrix
                        .where(~np.eye(len(num_cols), dtype=bool))
                        .stack()
                        .reset_index()
                    )
                    corr_pairs.columns = ["Col1", "Col2", "Correlation"]
                    corr_pairs = corr_pairs.sort_values(
                        "Correlation", ascending=False
                    )
                    corr_pairs = corr_pairs[
                        corr_pairs["Col1"] < corr_pairs["Col2"]
                    ]

                    pdf.set_fill_color(0, 100, 200)
                    pdf.set_text_color(255, 255, 255)
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.cell(70, 8, "Column 1",    border=1, fill=True)
                    pdf.cell(70, 8, "Column 2",    border=1, fill=True)
                    pdf.cell(50, 8, "Correlation", border=1,
                             fill=True, align="C")
                    pdf.ln()

                    for _, row in corr_pairs.head(6).iterrows():
                        corr_val = round(row["Correlation"], 3)
                        if corr_val >= 0.7:
                            pdf.set_fill_color(255, 235, 235)
                        elif corr_val >= 0.4:
                            pdf.set_fill_color(255, 250, 220)
                        else:
                            pdf.set_fill_color(240, 255, 240)
                        pdf.set_text_color(30, 30, 30)
                        pdf.set_font("Helvetica", "", 9)
                        pdf.cell(70, 7, str(row["Col1"]), border=1, fill=True)
                        pdf.cell(70, 7, str(row["Col2"]), border=1, fill=True)
                        pdf.cell(50, 7, str(corr_val),   border=1,
                                 fill=True, align="C")
                        pdf.ln()

                    pdf.ln(5)

                # ── Section 5: AI Narrative ──────────────
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(0, 100, 200)
                pdf.cell(0, 10, "5. AI Analyst Narrative", ln=True)
                pdf.set_draw_color(0, 100, 200)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(3)

                narrative = clean_text(st.session_state["narrative"])

                for line in narrative.split("\n"):
                    line = line.strip()
                    if not line:
                        pdf.ln(3)
                        continue
                    if re.match(r'^\d+\.', line):
                        pdf.set_font("Helvetica", "B", 11)
                        pdf.set_text_color(0, 100, 200)
                        pdf.cell(0, 8, line, ln=True)
                        pdf.set_draw_color(200, 220, 255)
                        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                        pdf.ln(1)
                    elif line.startswith("-"):
                        pdf.set_font("Helvetica", "", 10)
                        pdf.set_text_color(50, 50, 50)
                        pdf.set_x(15)
                        pdf.multi_cell(0, 6, line)
                    else:
                        pdf.set_font("Helvetica", "", 10)
                        pdf.set_text_color(50, 50, 50)
                        pdf.multi_cell(0, 6, line)
                        pdf.ln(1)

                # ── Save & Download ──────────────────────
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    pdf.output(tmp.name)
                    tmp_path = tmp.name

                with open(tmp_path, "rb") as f:
                    pdf_bytes = f.read()

                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"AutoEDA_Report_{filename.replace('.csv', '')}.pdf",
                    mime="application/pdf"
                )

                st.success("✅ PDF ready! Click the button above to download.")
                os.unlink(tmp_path)