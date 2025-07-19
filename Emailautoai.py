# ✅ Full Agentic + Multi-Agent AutoML System with Chat + EDA Email Notifications with AI Insight + Single PDF

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
from matplotlib.backends.backend_pdf import PdfPages
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
import xgboost as xgb

# === Together AI ===
together_api_keys = [
    "your_api_key_1",
    "your_api_key_2"
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
    return f"Error: {response.text}"

def ask_data_scientist_agent(prompt):
    return ask_agent(f"[DATA SCIENTIST] {prompt}", "mistralai/Mistral-7B-Instruct-v0.1", key=0)

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

class AutoMLAgent:
    def __init__(self, X, y):
        self.X_raw = X.copy()
        self.X = pd.get_dummies(X)
        self.y = y
        self.classification = self._detect_task_type()
        self.models = self._load_models()
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = -np.inf
        self.best_info = {}
        self.results = []

    def _detect_task_type(self):
        return self.y.dtype == 'object' or len(np.unique(self.y)) <= 20

    def _load_models(self):
        return {
            "classification": {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Extra Trees": ExtraTreesClassifier(),
                "AdaBoost": AdaBoostClassifier(),
                "KNN": KNeighborsClassifier(),
                "SVC": SVC(),
                "Naive Bayes (Gaussian)": GaussianNB(),
                "Naive Bayes (Multinomial)": MultinomialNB(),
                "Naive Bayes (Complement)": ComplementNB(),
                "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
            },
            "regression": {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Extra Trees": ExtraTreesRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "KNN": KNeighborsRegressor(),
                "SVR": SVR(),
                "XGBoost": xgb.XGBRegressor(),
                "Polynomial Linear Regression": make_pipeline(PolynomialFeatures(2), LinearRegression())
            }
        }["classification" if self.classification else "regression"]

    def run(self):
        for test_size in [0.1, 0.2, 0.3]:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)
            if self.classification and len(np.unique(y_train)) > 2:
                sampler = SMOTE()
                X_train, y_train = sampler.fit_resample(X_train, y_train)

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            for name, model in self.models.items():
                try:
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    score = accuracy_score(y_test, preds) if self.classification else r2_score(y_test, preds)
                    info = {
                        "Model": name,
                        "Score": round(score, 4),
                        "Test Size": test_size,
                        "Type": "Classification" if self.classification else "Regression"
                    }
                    self.results.append(info)
                    if score > self.best_score:
                        self.best_score = score
                        self.best_model = model
                        self.best_info = info
                except Exception:
                    continue

        return pd.DataFrame(self.results).sort_values(by="Score", ascending=False), self.best_info

    def save_best_model(self):
        with open("best_model.pkl", "wb") as f:
            pickle.dump(self.best_model, f)

st.set_page_config(page_title="Agentic AutoML AI", layout="wide")
st.title("🤖 Multi-Agent AutoML System with Email Intelligence")

uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    pdf_report_path = "eda_report.pdf"
    with PdfPages(pdf_report_path) as pdf:

        st.subheader("📉 Basic EDA")
        st.write(df.describe())
        st.write("Missing Values:")
        st.write(df.isnull().sum())

        fig, ax = plt.subplots()
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
        ax.set_title("Missing Data Visualization")
        pdf.savefig(fig)

        col1, col2 = st.columns([2, 3])
        with col1:
            st.pyplot(fig)
        with col2:
            summary = ask_data_scientist_agent(
                f"This is a missing value heatmap. Summary of dataset:\n{df.isnull().sum().to_string()}"
            )
            st.markdown("### 🧠 AI Insight on Missing Values")
            st.write(summary)

        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include='object').columns

        if not num_cols.empty:
            st.markdown("### 🔢 Numeric Feature Distributions")
            for col in num_cols:
                fig, ax = plt.subplots()
                df[col].hist(ax=ax, bins=20, color='skyblue', edgecolor='black')
                ax.set_title(f"Histogram of {col}")
                pdf.savefig(fig)
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.pyplot(fig)
                with col2:
                    summary = ask_data_scientist_agent(
                        f"Analyze this histogram for '{col}'.\nStatistics:\n{df[col].describe()}"
                    )
                    st.markdown(f"### 🧠 AI Insight on {col}")
                    st.write(summary)

            st.markdown("### 🧮 Box Plots (Outlier Detection)")
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.boxplot(data=df, x=col, ax=ax, color='lightcoral')
                ax.set_title(f"Box Plot of {col}")
                pdf.savefig(fig)
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.pyplot(fig)
                with col2:
                    summary = ask_data_scientist_agent(
                        f"Analyze outliers for '{col}' using this boxplot.\nStats:\n{df[col].describe()}"
                    )
                    st.markdown(f"### 🧠 AI Insight on {col} (Outliers)")
                    st.write(summary)

        if not cat_cols.empty:
            st.markdown("### 🗾 Categorical Feature Breakdown")
            for col in cat_cols:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar', ax=ax, color='lightgreen')
                ax.set_title(f"Bar Chart of {col}")
                pdf.savefig(fig)
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.pyplot(fig)
                with col2:
                    summary = ask_data_scientist_agent(
                        f"Analyze the frequency distribution of '{col}'.\nCounts:\n{df[col].value_counts()}"
                    )
                    st.markdown(f"### 🧠 AI Insight on {col} (Bar Chart)")
                    st.write(summary)

                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=90)
                ax.set_ylabel("")
                ax.set_title(f"Pie Chart of {col}")
                pdf.savefig(fig)
                col1, col2 = st.columns([2, 3])
                with col1:
                    st.pyplot(fig)
                with col2:
                    summary = ask_data_scientist_agent(
                        f"Interpret the pie chart for '{col}' category distribution."
                    )
                    st.markdown(f"### 🧠 AI Insight on {col} (Pie Chart)")
                    st.write(summary)

    if client_email:
        eda_summary = """
Dear Client,

Our system has completed the initial EDA analysis. The full PDF report is attached.

Please confirm if you'd like us to proceed with model training.

Regards,
Akash
        """
        send_email_report("Initial EDA PDF Report", eda_summary, client_email, [pdf_report_path])
        st.info("📧 PDF EDA report emailed to client.")
