from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model_utils import build_preprocessor, save_model, load_model

st.set_page_config(
    page_title="Movie Genre Classification Studio",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Genre Deep Learning Studio")
st.markdown("Multi-class Deep Neural Network (DNN) classifying movies into **Action**, **Comedy**, **Drama**, and **Thriller** based on Budget, Runtime, Rating, Release Year, and Language.")

DATA_FILE = Path(__file__).resolve().parent / "movie_genre.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)

df = load_data()

tab1, tab2, tab3 = st.tabs(["📊 Dataset Explorer", "⚡ Neural Network Studio", "🔮 Live Inference"])

# TAB 1
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Classes", df["Genre"].nunique())
    with col4:
        st.metric("Target", "Genre")

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Genre Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Genre", palette="magma", ax=ax)
        st.pyplot(fig)
    with col_b:
        st.subheader("Budget vs. Rating by Genre")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="Budget_Million", y="Rating", hue="Genre", palette="Set1", ax=ax2)
        st.pyplot(fig2)

# TAB 2
with tab2:
    st.subheader("Interactive Model Training")
    c1, c2, c3 = st.columns(3)
    with c1:
        l1 = st.slider("Layer 1 Neurons", 32, 512, 256, step=32)
        l2 = st.slider("Layer 2 Neurons", 16, 256, 128, step=16)
        l3 = st.slider("Layer 3 Neurons", 0, 128, 64, step=16)
    with c2:
        activation = st.selectbox("Activation", ["relu", "tanh", "logistic"])
        lr = st.select_slider("Learning Rate", options=[0.0005, 0.001, 0.005, 0.01], value=0.001)
        epochs = st.slider("Epochs", 50, 500, 250, step=25)
    with c3:
        test_size = st.slider("Test Split Ratio", 0.1, 0.4, 0.2, step=0.05)
        early_stop = st.checkbox("Early Stopping", value=True)

    layers = [l1, l2] if l3 == 0 else [l1, l2, l3]

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training Deep Neural Network..."):
            X = df.drop(columns=["Genre"])
            y_raw = df["Genre"]
            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            class_names = list(le.classes_)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            preprocessor, _, _ = build_preprocessor(X_train)
            X_train_proc = preprocessor.fit_transform(X_train)
            X_test_proc = preprocessor.transform(X_test)

            model = MLPClassifier(
                hidden_layer_sizes=tuple(layers),
                activation=activation,
                learning_rate_init=lr,
                max_iter=epochs,
                early_stopping=early_stop,
                random_state=42
            )
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_test_proc)
            acc = accuracy_score(y_test, y_pred)

            st.success(f"Training Complete! Test Accuracy: {acc*100:.2f}%")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
                st.pyplot(fig_cm)
            with col_res2:
                st.subheader("Loss Curve")
                fig_l, ax_l = plt.subplots(figsize=(6, 4))
                ax_l.plot(model.loss_curve_, color="#2563eb", lw=2)
                ax_l.set_xlabel("Epochs")
                ax_l.set_ylabel("Loss")
                st.pyplot(fig_l)

# TAB 3
with tab3:
    st.subheader("Live Prediction Playground")
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        in_budget = st.number_input("Budget ($ Millions)", min_value=1.0, max_value=500.0, value=75.0)
        in_runtime = st.number_input("Runtime (Minutes)", min_value=30, max_value=300, value=120)
    with col_in2:
        in_rating = st.slider("Rating (1-10)", min_value=1.0, max_value=10.0, value=7.5, step=0.1)
        in_year = st.slider("Release Year", min_value=1990, max_value=2026, value=2020)
    with col_in3:
        in_lang = st.selectbox("Language", ["English", "Hindi", "Spanish", "French"])

    if st.button("🔮 Predict Genre", type="primary"):
        sample_df = pd.DataFrame([{
            "Budget_Million": in_budget,
            "Runtime_Minutes": in_runtime,
            "Rating": in_rating,
            "Release_Year": in_year,
            "Language": in_lang
        }])

        X = df.drop(columns=["Genre"])
        y_raw = df["Genre"]
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        class_names = list(le.classes_)

        preprocessor, _, _ = build_preprocessor(X)
        X_proc = preprocessor.fit_transform(X)
        model = MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=200, random_state=42)
        model.fit(X_proc, y)

        sample_proc = preprocessor.transform(sample_df)
        pred_idx = model.predict(sample_proc)[0]
        pred_genre = class_names[pred_idx]
        probs = model.predict_proba(sample_proc)[0]

        st.success(f"🎯 **Predicted Genre:** `{pred_genre}`")

        prob_df = pd.DataFrame({"Genre": class_names, "Probability": probs})
        fig_p, ax_p = plt.subplots(figsize=(6, 3))
        sns.barplot(data=prob_df, x="Probability", y="Genre", palette="viridis", ax=ax_p)
        ax_p.set_xlim(0, 1.0)
        st.pyplot(fig_p)
