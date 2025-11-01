
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
from io import BytesIO

# Custom CSS for unique styling
st.markdown("""
<style>
    .stApp {
        background-color: #f9f9f9;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .header {
        font-size: 30px;
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        color: #555;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown('<p class="header">Automated Data Analysis and Machine Learning Pipeline</p>', unsafe_allow_html=True)

# Tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["Data Upload & Display", "Data Cleaning & Preprocessing", "Exploratory Data Analysis (EDA)", "Machine Learning"])

# Tab 1: Data Upload and Display
with tab1:
    st.markdown('<p class="subheader">1. Data Upload and Display</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"], help="Supported formats: CSV, Excel")

    if uploaded_file:
        # Load the data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success("Dataset successfully uploaded!")
        st.write("Uploaded Data Preview:")
        st.dataframe(df.head())

# Tab 2: Data Cleaning and Preprocessing
with tab2:
    st.markdown('<p class="subheader">2. Data Cleaning and Preprocessing</p>', unsafe_allow_html=True)

    if 'df' in locals():
        with st.expander("Missing Value Handling"):
            if df.isnull().sum().sum() > 0:
                missing_cols = df.columns[df.isnull().any()].tolist()
                st.warning(f"Columns with missing values: {missing_cols}")
                for col in missing_cols:
                    method = st.selectbox(f"Select handling method for column '{col}'", ["Mean Imputation", "Median Imputation", "Delete Rows"])
                    if method == "Mean Imputation":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == "Median Imputation":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif method == "Delete Rows":
                        df.dropna(subset=[col], inplace=True)
                st.info("Missing values handled successfully!")

        with st.expander("Duplicate Removal"):
            if st.checkbox("Remove Duplicate Rows"):
                df.drop_duplicates(inplace=True)
                st.info("Duplicate rows removed successfully!")

        with st.expander("Outlier Detection and Handling"):
            if st.checkbox("Detect and Handle Outliers (using IQR)"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                st.info("Outliers handled successfully!")

        with st.expander("Text Cleaning"):
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in text_cols:
                if st.checkbox(f"Clean Text in Column '{col}'"):
                    df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)).lower())
                    st.info(f"Text cleaned in column '{col}'!")

        with st.expander("Feature Scaling"):
            scaling_method = st.selectbox("Select Feature Scaling Method", ["None", "Standardization", "Normalization"])
            if scaling_method != "None":
                scaler = StandardScaler() if scaling_method == "Standardization" else MinMaxScaler()
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                st.info("Feature scaling applied successfully!")

        with st.expander("Encoding Categorical Variables"):
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            if not cat_cols:
                st.warning("No categorical columns found in the dataset.")
            else:
                encoding_method = st.selectbox(
                    "Select encoding method for categorical variables",
                    ["None", "One-Hot Encoding", "Label Encoding"]
                    )
                if encoding_method == "One-Hot Encoding":
                    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                    st.success("One-Hot Encoding applied to all categorical columns.")
                elif encoding_method == "Label Encoding":
                    if df[cat_cols].isnull().sum().sum() > 0:
                        st.warning("Some categorical columns contain missing values. Handle them before encoding.")
                    else:
                        le = LabelEncoder()
                        for col in cat_cols:
                            df[col] = le.fit_transform(df[col])
                            st.success("Label Encoding applied to all categorical columns.")

                else:
                    st.info("No encoding applied.")

                


# Tab 3: Exploratory Data Analysis (EDA)
with tab3:
    st.markdown('<p class="subheader">3. Exploratory Data Analysis (EDA)</p>', unsafe_allow_html=True)

    if 'df' in locals():
        with st.expander("Summary Statistics"):
            st.write(df.describe())

        with st.expander("Interactive Visualizations"):
            plot_type = st.selectbox("Select Plot Type", ["Histogram", "Scatter Plot", "Box Plot", "Correlation Matrix", "Pair Plot", "Bar Chart"])
            if plot_type == "Histogram":
                column = st.selectbox("Select Column for Histogram", df.columns)
                fig, ax = plt.subplots()
                ax.hist(df[column], bins=20, color="#4CAF50")
                st.pyplot(fig)
            elif plot_type == "Scatter Plot":
                x_col = st.selectbox("Select X-axis Column", df.columns)
                y_col = st.selectbox("Select Y-axis Column", df.columns)
                fig, ax = plt.subplots()
                ax.scatter(df[x_col], df[y_col], color="#FF5733")
                st.pyplot(fig)
            elif plot_type == "Box Plot":
                column = st.selectbox("Select Column for Box Plot", df.columns)
                fig, ax = plt.subplots()
                sns.boxplot(x=df[column], ax=ax, palette="Set2")
                st.pyplot(fig)
            elif plot_type == "Correlation Matrix":
                corr_matrix = df.corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
            elif plot_type == "Pair Plot":
                fig = sns.pairplot(df, palette="husl")
                st.pyplot(fig)
            elif plot_type == "Bar Chart":
                column = st.selectbox("Select Column for Bar Chart", df.columns)
                fig, ax = plt.subplots()
                df[column].value_counts().plot(kind='bar', ax=ax, color="#3399FF")
                st.pyplot(fig)

# Tab 4: Machine Learning
with tab4:
    st.markdown('<p class="subheader">4. Machine Learning Model Selection and Training</p>', unsafe_allow_html=True)

    if 'df' in globals() and df is not None:
        target_column = st.selectbox("Select Target Column", df.columns)
        
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression"])
            models = {
                "Classification": {
                    "Logistic Regression": LogisticRegression(),
                    "Random Forest Classifier": RandomForestClassifier(),
                    "SVM": SVC(),
                    "Naive Bayes": GaussianNB()
                },
                "Regression": {
                    "Linear Regression": LinearRegression(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "SVR": SVR()
                }
            }
            
            selected_model_name = st.selectbox("Select Model", list(models[problem_type].keys()))
            selected_model = models[problem_type][selected_model_name]
            
            if st.button("Train Model 🚀"):
                with st.spinner("Training the model..."):
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Feature Scaling for specific models
                    if isinstance(selected_model, (LogisticRegression, SVC, SVR)):
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    selected_model.fit(X_train, y_train)
                    y_pred = selected_model.predict(X_test)
                
                st.success("Model training completed successfully!")
                st.subheader("Model Evaluation Metrics")
                
                if problem_type == "Classification":
                    st.write("Accuracy:", accuracy_score(y_test, y_pred))
                    st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
                    st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
                    st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
                    
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    st.pyplot(fig)
                    
                    st.subheader("Classification Report")
                    st.text(classification_report(y_test, y_pred))
                    
                else:
                    st.write("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
                    st.write("R-squared:", r2_score(y_test, y_pred))
                    
                    st.subheader("Predictions vs Actual Values")
                    fig, ax = plt.subplots()
                    sns.scatterplot(x=y_test, y=y_pred, color="#3399FF", ax=ax)
                    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                    ax.set_xlabel("Actual")
                    ax.set_ylabel("Predicted")
                    st.pyplot(fig)
# Ensure model is trained before saving
if 'selected_model' in locals() and selected_model is not None:
    model_filename = f"{selected_model_name.replace(' ', '_').lower()}_model.pkl"
    
    if st.button("Save Model 🛠"):
        joblib.dump(selected_model, model_filename)
        st.success(f"Model saved as {model_filename}!")
        
        # Provide a download link
        with open(model_filename, "rb") as file:
            btn = st.download_button(
                label="Download Model 📥",
                data=file.read(),
                file_name=model_filename,
                mime="application/octet-stream"
            )

# Footer
st.markdown("---")

st.markdown("<p style='text-align: center; color: #777;'>Developed  ❤️ by  venu</p>", unsafe_allow_html=True)
