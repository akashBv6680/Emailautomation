# === Required Imports (same as before) ===
import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import smtplib
import seaborn as sns
import matplotlib.pyplot as plt
import os
from email.message import EmailMessage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from imblearn.over_sampling import SMOTE
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import xgboost as xgb
import textwrap

# === Together AI Keys ===
together_api_keys = [
    "tgp_v1_ecSsk1__FlO2mB_gAaaP2i-Affa6Dv8OCVngkWzBJUY",
    "tgp_v1_4hJBRX0XDlwnw_hhUnhP0e_lpI-u92Xhnqny2QIDAIM"
]

client_email = st.sidebar.text_input("Enter Client Email")

def ask_agent(prompt, model, key=0):
    response = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        headers={"Authorization": f"Bearer {together_api_keys[key]}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    return "AI Insight unavailable."

def simplify_insight(raw_text):
    bullets = [line.strip("-• ") for line in raw_text.split("\n") if line.strip() and not any(x in line.lower() for x in ["chart is", "represents", "graph", "x-axis", "y-axis", "category"])]
    simple = bullets[:3] if len(bullets) >= 3 else bullets
    return '\n'.join([f"• {line}" for line in simple])

def ask_data_scientist_agent(prompt):
    return ask_agent(f"[DATA SCIENTIST] {prompt} Use only simple English. Provide only 3 short, clear bullet points as the result.", "mistralai/Mistral-7B-Instruct-v0.1", key=0)

def send_email_report(subject, body, to, attachment_paths=None):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = st.secrets["EMAIL_ADDRESS"]
    msg['To'] = to
    msg.set_content(body)

    if attachment_paths:
        for path in attachment_paths:
            with open(path, 'rb') as f:
                file_data = f.read()
            msg.add_attachment(file_data, maintype='application', subtype='pdf', filename=os.path.basename(path))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(st.secrets["EMAIL_ADDRESS"], st.secrets["EMAIL_PASSWORD"])
        smtp.send_message(msg)

# === Streamlit UI ===
st.set_page_config(page_title="Agentic AutoML AI", layout="wide")
st.title("🤖 Multi-Agent AutoML System with Email Intelligence")

uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📉 Basic EDA")
    st.write(df.describe())
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    insights = []
    pdf_path = "eda_report.pdf"

    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        ax.set_title("Missing Data Visualization")
        pdf.savefig(fig)
        st.pyplot(fig)
        plt.close()
        raw_insight = ask_data_scientist_agent("Explain the missing value heatmap for a client.")
        insight = simplify_insight(raw_insight)
        insights.append(("Missing Data Visualization", insight))
        st.markdown(f"**AI Insight:**\n{insight}")

        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include='object').columns

        for col in num_cols:
            for chart_type in ["hist", "box"]:
                fig = plt.figure(figsize=(11, 5))
                gs = GridSpec(1, 2, width_ratios=[2, 1])
                ax1 = fig.add_subplot(gs[0])
                if chart_type == "hist":
                    df[col].hist(ax=ax1, bins=20, color='skyblue', edgecolor='black')
                    ax1.set_title(f"Histogram of {col}")
                    summary = df[col].describe().to_string()
                    raw_insight = ask_data_scientist_agent(f"Explain histogram of '{col}' in simple words. Stats:\n{summary}")
                else:
                    sns.boxplot(data=df, x=col, ax=ax1, color='lightcoral')
                    ax1.set_title(f"Box Plot of {col}")
                    raw_insight = ask_data_scientist_agent(f"What does the box plot of '{col}' show?")

                insight = simplify_insight(raw_insight)
                insights.append((f"{chart_type.title()} of {col}", insight))
                ax2 = fig.add_subplot(gs[1])
                ax2.axis('off')
                ax2.text(0, 1, insight, wrap=True, fontsize=10, verticalalignment='top')
                pdf.savefig(fig)
                plt.close()
                st.pyplot(fig)
                st.markdown(f"**{col} {chart_type.title()} Insight:**\n{insight}")

        for col in cat_cols:
            for chart_type in ["bar", "pie"]:
                fig = plt.figure(figsize=(11, 5))
                gs = GridSpec(1, 2, width_ratios=[2, 1])
                ax1 = fig.add_subplot(gs[0])
                if chart_type == "bar":
                    df[col].value_counts().plot(kind='bar', ax=ax1, color='lightgreen')
                    ax1.set_title(f"Bar Chart of {col}")
                    raw_insight = ask_data_scientist_agent(f"Explain bar chart of column '{col}' in 3 short simple points.")
                else:
                    df[col].value_counts().plot(kind='pie', ax=ax1, autopct='%1.1f%%', startangle=90)
                    ax1.set_ylabel("")
                    ax1.set_title(f"Pie Chart of {col}")
                    raw_insight = ask_data_scientist_agent(f"Explain pie chart of column '{col}' for a client.")

                insight = simplify_insight(raw_insight)
                insights.append((f"{chart_type.title()} of {col}", insight))
                ax2 = fig.add_subplot(gs[1])
                ax2.axis('off')
                ax2.text(0, 1, insight, wrap=True, fontsize=10, verticalalignment='top')
                pdf.savefig(fig)
                plt.close()
                st.pyplot(fig)
                st.markdown(f"**{col} {chart_type.title()} Insight:**\n{insight}")

        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        summary_text = "\n\n".join([f"{insight}" for _, insight in insights])
        ax.text(0.01, 0.99, "📄 Summary of Visual Insights:\n\n" + summary_text, fontsize=10, va='top', wrap=True)
        pdf.savefig(fig)
        plt.close()

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download Full AI Insights Report (PDF)", f, file_name="EDA_AI_Insights_Report.pdf")



    problem_detected = df.isnull().sum().any() or df.select_dtypes(include=np.number).apply(lambda x: ((x - x.mean())/x.std()).abs().gt(3).sum()).sum() > 0

    if problem_detected and client_email:
        eda_summary = """
Dear Client,

Our system has completed the initial analysis of your dataset. Here are the key observations:

- ❗ Potential data quality issues found (missing values or outliers)
- 🧹 Visuals and insights attached in the PDF report

Please confirm if you'd like us to proceed with data cleaning and model training.

Regards,
Akash
        """
        send_email_report("Initial Data Quality Report", eda_summary, client_email, [pdf_path])
        st.warning("Initial report emailed to client for confirmation before continuing.")

        proceed = st.checkbox("✅ Client confirmed. Proceed with model training?")
        if proceed:
            target = st.selectbox("🎯 Select Target Variable", df.columns)
            if target:
                X = df.drop(columns=[target])
                y = df[target]

                agent = AutoMLAgent(X, y)
                results_df, best_info = agent.run()

                st.subheader("🏆 Model Leaderboard")
                st.dataframe(results_df)

                agent.save_best_model()
                st.success(f"Best Model: {best_info['Model']} with score: {best_info['Score']}")

                model_summary = f"""
Dear Client,

The AutoML process is complete. Here are the results:

✅ Best Model: {best_info['Model']}
📈 Score: {best_info['Score']}
📊 Type: {best_info['Type']}
🔎 Test Size: {best_info['Test Size']}

Thank you for using our AI service.

Regards,
Akash
"""
                send_email_report("Final AutoML Model Report", model_summary, client_email, [pdf_path])
