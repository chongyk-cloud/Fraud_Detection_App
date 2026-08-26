import streamlit as st
import pandas as pd
import numpy as np
import joblib
import __main__
from sklearn.base import BaseEstimator, TransformerMixin

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
with tab2:
    st.write("Batch Auction Audit content will be built here next.")

with tab3:
    st.write("Model Hub content will be built here.")
