import streamlit as st
import pandas as pd
import numpy as np
import joblib
import __main__
import plotly.express as px  # <-- ADD THIS LINE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- 1. REQUIRED CLASS DEFINITION ---
class BoxplotWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, features_with_outliers=None):
        self.features_with_outliers = features_with_outliers or ["Bidder_Tendency", "Bidding_Ratio", "Winning_Ratio"]
        self.fences_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for feature in self.features_with_outliers:
            if feature in X_df.columns:
                q1 = X_df[feature].quantile(0.25)
                q3 = X_df[feature].quantile(0.75)
                iqr = q3 - q1
                self.fences_[feature] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy() if not isinstance(X, pd.DataFrame) else X.copy()
        for feature in self.features_with_outliers:
            if feature in self.fences_:
                lower, upper = self.fences_[feature]
                X_df[feature] = np.clip(X_df[feature], lower, upper)
        return X_df

# Map the class to __main__ so joblib can unpickle it correctly on the cloud
__main__.BoxplotWinsorizer = BoxplotWinsorizer

# --- 2. MODEL PATHS (LAZY LOADING) ---
# We ONLY store the file names here. We don't load them into memory yet!
MODEL_PATHS = {
    "Logistic Regression (Baseline)": 'logistic_regression_model.pkl',
    "Random Forest Classifier": 'random_forest_model.pkl',
    "Hist Gradient Boosting": 'hist_gradient_boosting_model.pkl'
}

# --- 3. PAGE SETUP ---
st.set_page_config(
    page_title="Auction Shield | Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# --- 4. HEADER & TITLE ---
st.title("🛡️ Auction Shield: Shill Bidding Detection System")
st.markdown("Real-time financial security and risk analytics portal for online auction integrity.")
st.divider()

# --- 5. TABS NAVIGATION ---
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Bidder Inspection",
    "📁 Batch Auction Audit",
    "📊 Model Hub"
])

# ==========================================
# MODULE 1: LIVE BIDDER INSPECTION (TAB 1)
# ==========================================
with tab1:
    st.subheader("Interactive Bidder Simulator")
    st.markdown("Adjust the behavioral metrics below to test the machine learning models in real-time.")
    
    # Model Selector
    selected_model_name = st.selectbox("Select Fraud Detection Algorithm:", list(MODEL_PATHS.keys()))
    
    st.divider()
    
    # Input Sliders (Organized in 3 columns for a clean UI)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bidder_tendency = st.slider("Bidder Tendency", 0.0, 1.0, 0.5)
        bidding_ratio = st.slider("Bidding Ratio", 0.0, 1.0, 0.2)
        successive_outbidding = st.slider("Successive Outbidding", 0.0, 1.0, 0.0)
        
    with col2:
        last_bidding = st.slider("Last Bidding", 0.0, 1.0, 0.0)
        auction_bids = st.slider("Auction Bids", 0.0, 10.0, 1.0)
        starting_price_average = st.slider("Starting Price Average", 0.0, 1.0, 0.0)
        
    with col3:
        early_bidding = st.slider("Early Bidding", 0.0, 1.0, 0.0)
        winning_ratio = st.slider("Winning Ratio", 0.0, 1.0, 0.0)
        auction_duration = st.slider("Auction Duration (Days)", 1, 10, 7)

    # Predict Button
    if st.button("Analyze Bidder Risk", type="primary", use_container_width=True):
        
        # LAZY LOAD THE MODEL HERE: It only loads the one you picked, saving RAM!
        with st.spinner(f"Loading {selected_model_name} and analyzing..."):
            active_model = joblib.load(MODEL_PATHS[selected_model_name])
            
            # Package inputs into a DataFrame matching the EXACT training data order
            input_data = pd.DataFrame([[
                bidder_tendency, 
                bidding_ratio, 
                successive_outbidding,
                last_bidding, 
                auction_bids, 
                starting_price_average, 
                early_bidding, 
                winning_ratio, 
                auction_duration
            ]], columns=[
                "Bidder_Tendency", 
                "Bidding_Ratio", 
                "Successive_Outbidding",
                "Last_Bidding", 
                "Auction_Bids", 
                "Starting_Price_Average", 
                "Early_Bidding", 
                "Winning_Ratio", 
                "Auction_Duration"
            ])
            
            # Make Prediction
            prediction = active_model.predict(input_data)[0]
            probability = active_model.predict_proba(input_data)[0][1]
        
        # Display Results
        st.divider()
        if prediction == 1:
            st.error(f"🚨 **FRAUDULENT BIDDER DETECTED (SHILL)**")
            st.warning(f"**Risk Confidence Score:** {probability * 100:.2f}% probability of shill activity.")
        else:
            st.success(f"✅ **LEGITIMATE BIDDER (NORMAL)**")
            st.info(f"**Risk Confidence Score:** {probability * 100:.2f}% probability of shill activity.")

# ==========================================
# MODULE 2 & 3: PLACEHOLDERS
# ==========================================
# ==========================================
# MODULE 2: BATCH AUCTION AUDIT (TAB 2)
# ==========================================
with tab2:
    st.subheader("📁 Batch Auction Audit")
    st.markdown("Run automated security scans across bulk auction logs to identify widespread fraudulent patterns.")
    
    col1, col2 = st.columns(2)
    with col1:
        data_source = st.radio("Select Data Source:", ["Use Default Dataset (Shill Bidding Dataset.csv)", "Upload New CSV"])
    with col2:
        # Give this a unique key so it doesn't conflict with Tab 1's selector
        batch_model_name = st.selectbox("Select Model for Batch Scan:", list(MODEL_PATHS.keys()), key="batch_model")

    uploaded_file = None
    if data_source == "Upload New CSV":
        uploaded_file = st.file_uploader("Upload Auction Data (CSV)", type=["csv"])

    if st.button("Run Batch Audit", type="primary", use_container_width=True):
        # 1. Load Data
        df = None
        try:
            if data_source == "Use Default Dataset (Shill Bidding Dataset.csv)":
                df = pd.read_csv("Shill Bidding Dataset.csv")
            else:
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                else:
                    st.warning("Please upload a CSV file first.")
        except Exception as e:
            st.error(f"Error loading data: {e}")

        # 2. Process Data if Loaded Successfully
        if df is not None:
            feature_cols = [
                "Bidder_Tendency", "Bidding_Ratio", "Successive_Outbidding",
                "Last_Bidding", "Auction_Bids", "Starting_Price_Average", 
                "Early_Bidding", "Winning_Ratio", "Auction_Duration"
            ]
            
            # Check if CSV has the correct columns
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                st.error(f"🚨 Missing required columns in dataset: {missing_cols}")
            else:
                with st.spinner(f"Scanning {len(df)} records using {batch_model_name}..."):
                    # Load Model & Predict
                    batch_model = joblib.load(MODEL_PATHS[batch_model_name])
                    X_batch = df[feature_cols]
                    predictions = batch_model.predict(X_batch)
                    
                    # Append results to the dataframe
                    results_df = df.copy()
                    results_df["Fraud_Prediction"] = predictions
                    results_df["Risk_Status"] = results_df["Fraud_Prediction"].apply(lambda x: "🚨 Shill" if x == 1 else "✅ Clean")
                    
                    # Calculate Metrics
                    total_bids = len(results_df)
                    flagged_shills = int(sum(predictions))
                    clean_bids = total_bids - flagged_shills
                    clean_rate = (clean_bids / total_bids) * 100 if total_bids > 0 else 0
                    
                    # Display Metrics
                    st.divider()
                    st.markdown("### 📊 Audit Summary")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Bids Scanned", f"{total_bids:,}")
                    m2.metric("🚨 Flagged Shills", f"{flagged_shills:,}")
                    m3.metric("✅ Clean Bids Rate", f"{clean_rate:.2f}%")
                    
                    # Display Data Table (Highlighting Fraud Rows)
                    st.markdown("### 📋 Detailed Audit Log")
                    
                    def highlight_fraud(row):
                        if row['Fraud_Prediction'] == 1:
                            return ['background-color: rgba(255, 75, 75, 0.2)'] * len(row)
                        return [''] * len(row)
                    
                    st.dataframe(results_df.style.apply(highlight_fraud, axis=1), use_container_width=True)

                    st.markdown("### 🗺️ Interactive Risk Topography")
                    # Create an interactive scatter plot
                    fig = px.scatter(
                        results_df,
                        x="Bidding_Ratio",
                        y="Successive_Outbidding",
                        color="Risk_Status",
                        color_discrete_map={"🚨 Shill": "#ff4b4b", "✅ Clean": "#21c354"},
                        title="Bidder Behavior Clustering",
                        hover_data=["Bidder_Tendency", "Auction_Bids", "Fraud_Prediction"]
                    )
                    
                    # Update layout for a sleek, dark-mode friendly look
                    fig.update_layout(xaxis_title="Bidding Ratio", yaxis_title="Successive Outbidding")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download Button for Flagged Data
                    flagged_df = results_df[results_df["Fraud_Prediction"] == 1]
                    if not flagged_df.empty:
                        csv_export = flagged_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Flagged Accounts Report (CSV)",
                            data=csv_export,
                            file_name="flagged_shill_bidders.csv",
                            mime="text/csv",
                            type="primary"
                        )
# ==========================================
# MODULE 3: MODEL PERFORMANCE HUB (TAB 3)
# ==========================================
with tab3:
    st.subheader("📊 Model Performance & Evaluation Hub")
    st.markdown("This dashboard evaluates the three machine learning algorithms live against the uploaded dataset.")
    
    try:
        # Load the dataset for evaluation
        df_eval = pd.read_csv("Shill Bidding Dataset.csv")
        
        feature_cols = [
            "Bidder_Tendency", "Bidding_Ratio", "Successive_Outbidding",
            "Last_Bidding", "Auction_Bids", "Starting_Price_Average", 
            "Early_Bidding", "Winning_Ratio", "Auction_Duration"
        ]
        
        # Prepare X (features) and y (target)
        X_eval = df_eval[feature_cols]
        # Note: 'Class' is the standard target column name for this dataset. 
        # If your dataset uses a different name (like 'Fraud' or 'Target'), change it here!
        y_eval = df_eval["Class"] 
        
        metrics_list = []
        
        with st.spinner("Calculating live performance metrics..."):
            for model_name, path in MODEL_PATHS.items():
                # Load each model one by one
                eval_model = joblib.load(path)
                
                # Generate predictions
                y_pred = eval_model.predict(X_eval)
                y_prob = eval_model.predict_proba(X_eval)[:, 1]
                
                # Calculate metrics
                acc = accuracy_score(y_eval, y_pred)
                prec = precision_score(y_eval, y_pred)
                rec = recall_score(y_eval, y_pred)
                f1 = f1_score(y_eval, y_pred)
                roc = roc_auc_score(y_eval, y_prob)
                
                # Append to list
                metrics_list.append({
                    "Algorithm": model_name,
                    "Accuracy": f"{acc * 100:.2f}%",
                    "Precision (Fraud)": f"{prec * 100:.2f}%",
                    "Recall (Fraud)": f"{rec * 100:.2f}%",
                    "F1-Score": f"{f1 * 100:.2f}%",
                    "ROC-AUC": f"{roc:.3f}"
                })
        
        # Create and display the DataFrame
        metrics_df = pd.DataFrame(metrics_list)
        st.markdown("### 🏆 Live Comparative Metric Scoreboard")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        st.divider()
        
        # Analytical Insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🧠 Model Architecture Justification")
            st.info("""
            **Random Forest** and **Hist Gradient Boosting** significantly outperform the baseline Logistic Regression model. 
            Because the original dataset was highly imbalanced, combining these tree-based ensemble methods with **SMOTE** (Synthetic Minority Over-sampling Technique) allowed the algorithms to successfully learn the complex, non-linear behavioral patterns of shill bidders.
            """)
            
        with col2:
            st.markdown("### 🎯 Real-World Business Impact")
            st.success("""
            In financial security and fraud detection, **Recall** is the most critical metric. Missing a fraudulent bidder (False Negative) severely damages platform trust, which is far more dangerous than occasionally flagging a normal bidder for manual review (False Positive). 
            
            The model with the highest recall score is the recommended choice for protecting auction integrity.
            """)

    except Exception as e:
        st.error(f"Could not calculate live metrics. Error: {e}")
        st.info("Make sure 'Shill Bidding Dataset.csv' is in your repository and that the target variable column is named exactly 'Class'.")
