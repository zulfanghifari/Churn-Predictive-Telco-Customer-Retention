import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load model and explainer
@st.cache_resource
def load_model():
    with open('model_gbc.sav', 'rb') as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_explainer():
    with open('lime_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    return explainer

model = load_model()
explainer = load_explainer()

# Title
st.title("🎯 Customer Churn Prediction System")
st.markdown("---")

# Create three columns for feature input
col1, col2, col3 = st.columns(3)

# Column 1: Customer Demographics & Account Info
with col1:
    st.subheader("👤 Customer Information")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    dependents = st.selectbox("Dependents", options=['Yes', 'No'])
    
    st.subheader("💳 Billing Information")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.80, max_value=118.65, value=65.0, step=0.5)
    contract = st.selectbox("Contract Type", options=['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", options=['Yes', 'No'])

# Column 2: Internet Services
with col2:
    st.subheader("🌐 Internet Service")
    internet_service = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'])
    
    # Show these only if internet service is not 'No'
    if internet_service == 'No':
        online_security = 'No internet service'
        online_backup = 'No internet service'
        device_protection = 'No internet service'
        tech_support = 'No internet service'
        st.info("Internet service options disabled (No internet service selected)")
    else:
        online_security = st.selectbox("Online Security", options=['Yes', 'No', 'No internet service'])
        online_backup = st.selectbox("Online Backup", options=['Yes', 'No', 'No internet service'])
        device_protection = st.selectbox("Device Protection", options=['Yes', 'No', 'No internet service'])
        tech_support = st.selectbox("Tech Support", options=['Yes', 'No', 'No internet service'])

# Column 3: Summary
with col3:
    st.subheader("📊 Input Summary")
    st.write(f"**Tenure:** {tenure} months")
    st.write(f"**Monthly Charges:** ${monthly_charges:.2f}")
    st.write(f"**Contract:** {contract}")
    st.write(f"**Internet Service:** {internet_service}")
    st.write(f"**Dependents:** {dependents}")
    st.write(f"**Paperless Billing:** {paperless_billing}")
    
    # Calculate estimated revenue
    if contract == 'Month-to-month':
        est_revenue = monthly_charges
    elif contract == 'One year':
        est_revenue = monthly_charges * 12
    else:  # Two year
        est_revenue = monthly_charges * 24
    
    st.metric("Estimated Contract Value", f"${est_revenue:.2f}")

st.markdown("---")

# Centered predict button
col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)

# Prediction and LIME explanation
if predict_button:
    # Create input dataframe with correct feature order
    input_data = pd.DataFrame({
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'Dependents': [dependents],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'InternetService': [internet_service],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing]
    })
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]
    
    # Display prediction results
    st.markdown("---")
    st.subheader("📈 Prediction Results")
    
    result_col1, result_col2, result_col3 = st.columns(3)
    
    with result_col1:
        if prediction == 1:
            st.error("⚠️ **HIGH RISK** - Customer likely to churn")
        else:
            st.success("✅ **LOW RISK** - Customer likely to stay")
    
    with result_col2:
        st.metric("Churn Probability", f"{prediction_proba[1]*100:.1f}%")
    
    with result_col3:
        st.metric("Retention Probability", f"{prediction_proba[0]*100:.1f}%")
    
    # LIME Explanation
    st.markdown("---")
    st.subheader("🔍 LIME Explanation - Feature Importance")
    st.write("Understanding which features contribute most to this prediction:")
    
    with st.spinner("Generating LIME explanation..."):
        try:
            # Transform input data using the preprocessing step
            transformer = model.named_steps['transformer']
            transformed_data = transformer.transform(input_data)
            
            # Convert to numpy array if needed
            if hasattr(transformed_data, 'toarray'):
                transformed_data = transformed_data.toarray()
            
            # Create prediction function for LIME
            def predict_fn(data):
                # The data coming from LIME is already transformed
                # We need to use only the model step, not the full pipeline
                resampler = model.named_steps['resampler']
                clf = model.named_steps['model']
                
                # For prediction, we bypass resampler (it's only for training)
                return clf.predict_proba(data)
            
            # Generate LIME explanation
            exp = explainer.explain_instance(
                transformed_data[0], 
                predict_fn,
                num_features=10
            )
            
            # Create and display the plot
            fig = exp.as_pyplot_figure()
            fig.tight_layout()
            st.pyplot(fig)
            
            # Display feature contributions as a table
            st.markdown("#### Feature Contributions")
            feature_importance = exp.as_list()
            importance_df = pd.DataFrame(feature_importance, columns=['Feature', 'Impact'])
            importance_df['Impact'] = importance_df['Impact'].round(4)
            
            # Color code positive and negative impacts
            def color_impact(val):
                if val > 0:
                    return 'background-color: #ffcccc'
                else:
                    return 'background-color: #ccffcc'
            
            styled_df = importance_df.style.applymap(color_impact, subset=['Impact'])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
            st.info("🔴 Red features increase churn probability | 🟢 Green features decrease churn probability")
            
        except Exception as e:
            st.error(f"Error generating LIME explanation: {str(e)}")
            st.write("Debug info:")
            st.write(f"Transformed data shape: {transformed_data.shape}")
            st.write(f"Model steps: {list(model.named_steps.keys())}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>💡 This model helps identify customers at risk of churning based on their service usage and billing patterns.</p>
    </div>
    """,
    unsafe_allow_html=True
)