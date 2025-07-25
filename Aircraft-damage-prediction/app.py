# streamlit_aircraft_damage_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Aircraft Damage Prediction System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# App Configuration
CONFIG = {
    'model_path': 'aircraft_damage_model.pkl',
    'preprocessed_data_path': 'AviationData_preprocessed.csv',
    'plots_directory': 'visualization_plots',
    'summary_report_path': 'visualization_plots/model_summary_report.json'
}

@st.cache_data
def load_model_package():
    """Load the trained model package with caching."""
    try:
        model_package = joblib.load(CONFIG['model_path'])
        return model_package
    except FileNotFoundError:
        st.error(f"Model file not found: {CONFIG['model_path']}")
        st.error("Please run the model training script first!")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample preprocessed data for reference."""
    try:
        data = pd.read_csv(CONFIG['preprocessed_data_path'])
        return data
    except FileNotFoundError:
        st.warning("Sample data not found. Manual input only.")
        return None
    except Exception as e:
        st.warning(f"Error loading sample data: {str(e)}")
        return None

@st.cache_data
def load_model_summary():
    """Load model performance summary."""
    try:
        with open(CONFIG['summary_report_path'], 'r') as f:
            summary = json.load(f)
        return summary
    except FileNotFoundError:
        st.warning("Model summary not found.")
        return None
    except Exception as e:
        st.warning(f"Error loading model summary: {str(e)}")
        return None

def create_feature_input_form(model_package, sample_data):
    """Create an interactive form for feature input."""
    st.markdown('<div class="sub-header">🔧 Aircraft & Incident Details</div>', unsafe_allow_html=True)
    
    feature_names = model_package['feature_names']
    
    # Create input widgets based on feature names and sample data
    user_inputs = {}
    
    # Organize inputs into columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Time & Location**")
        
        # Time-based features
        if 'Event_Year' in feature_names:
            user_inputs['Event_Year'] = st.slider(
                "Event Year", 
                min_value=1980, 
                max_value=2024, 
                value=2020,
                help="Year when the incident occurred"
            )
        
        if 'Event_Month' in feature_names:
            user_inputs['Event_Month'] = st.selectbox(
                "Event Month",
                options=list(range(1, 13)),
                index=5,
                help="Month when the incident occurred"
            )
        
        if 'Event_DayOfWeek' in feature_names:
            user_inputs['Event_DayOfWeek'] = st.selectbox(
                "Day of Week",
                options=list(range(7)),
                format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
                index=2,
                help="Day of the week (0=Monday, 6=Sunday)"
            )
        
        if 'Event_Quarter' in feature_names:
            user_inputs['Event_Quarter'] = st.selectbox(
                "Quarter",
                options=[1, 2, 3, 4],
                index=1,
                help="Quarter of the year"
            )
    
    with col2:
        st.markdown("**Aircraft Information**")
        
        # Aircraft-related features
        if 'Amateur_Built' in feature_names:
            user_inputs['Amateur_Built'] = st.selectbox(
                "Amateur Built Aircraft",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=0,
                help="Is this an amateur-built aircraft?"
            )
        
        if 'Number_of_Engines' in feature_names:
            user_inputs['Number_of_Engines'] = st.selectbox(
                "Number of Engines",
                options=[1, 2, 3, 4],
                index=0,
                help="Number of engines on the aircraft"
            )
        
        if 'Airport_Related' in feature_names:
            user_inputs['Airport_Related'] = st.selectbox(
                "Airport Related",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes",
                index=1,
                help="Did the incident occur at or near an airport?"
            )
        
        # Coordinates (if available)
        if 'Latitude' in feature_names:
            user_inputs['Latitude'] = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=39.0,
                format="%.6f",
                help="Latitude coordinate of the incident"
            )
        
        if 'Longitude' in feature_names:
            user_inputs['Longitude'] = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-98.0,
                format="%.6f",
                help="Longitude coordinate of the incident"
            )
    
    with col3:
        st.markdown("**Injury Information**")
        
        # Injury-related features
        injury_features = ['Total_Fatal_Injuries', 'Total_Serious_Injuries', 
                          'Total_Minor_Injuries', 'Total_Uninjured']
        
        for feature in injury_features:
            if feature in feature_names:
                user_inputs[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=0,
                    max_value=500,
                    value=0,
                    help=f"Number of {feature.replace('Total_', '').replace('_', ' ').lower()}"
                )
        
        # Calculate derived features
        if all(f in user_inputs for f in injury_features):
            user_inputs['Total_Injuries'] = (user_inputs.get('Total_Fatal_Injuries', 0) + 
                                           user_inputs.get('Total_Serious_Injuries', 0) + 
                                           user_inputs.get('Total_Minor_Injuries', 0))
            
            user_inputs['Total_Occupants'] = (user_inputs['Total_Injuries'] + 
                                            user_inputs.get('Total_Uninjured', 0))
            
            user_inputs['Has_Fatal'] = 1 if user_inputs.get('Total_Fatal_Injuries', 0) > 0 else 0
            user_inputs['Has_Serious'] = 1 if user_inputs.get('Total_Serious_Injuries', 0) > 0 else 0
            user_inputs['Has_Minor'] = 1 if user_inputs.get('Total_Minor_Injuries', 0) > 0 else 0
    
    # Additional features with default values
    st.markdown("**Additional Parameters**")
    
    # Add any remaining features with default values
    remaining_features = [f for f in feature_names if f not in user_inputs]
    
    if remaining_features:
        with st.expander("Advanced Settings (Optional)", expanded=False):
            cols = st.columns(3)
            for i, feature in enumerate(remaining_features[:15]):  # Limit to first 15 for UI
                with cols[i % 3]:
                    if sample_data is not None and feature in sample_data.columns:
                        default_val = float(sample_data[feature].median())
                        user_inputs[feature] = st.number_input(
                            feature,
                            value=default_val,
                            format="%.4f",
                            help=f"Default: {default_val:.4f}"
                        )
                    else:
                        user_inputs[feature] = st.number_input(
                            feature,
                            value=0.0,
                            format="%.4f"
                        )
    
    # Fill any missing features with defaults
    for feature in feature_names:
        if feature not in user_inputs:
            if sample_data is not None and feature in sample_data.columns:
                user_inputs[feature] = float(sample_data[feature].median())
            else:
                user_inputs[feature] = 0.0
    
    return user_inputs

def make_prediction(model_package, user_inputs):
    """Make prediction using the model."""
    model = model_package['model']
    le_target = model_package['label_encoder']
    feature_names = model_package['feature_names']
    
    # Create input DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Ensure all required features are present and in the right order
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0] if hasattr(model, 'predict_proba') else None
    
    # Convert back to original labels
    predicted_label = le_target.inverse_transform([prediction])[0]
    
    return predicted_label, prediction_proba, le_target.classes_

def display_prediction_results(predicted_label, prediction_proba, classes):
    """Display prediction results with visualizations."""
    st.markdown('<div class="sub-header">Prediction Results</div>', unsafe_allow_html=True)
    
    # Main prediction display
    st.markdown(f'''
    <div class="prediction-box">
        <h2>Predicted Aircraft Damage Level</h2>
        <h1 style="font-size: 3rem; margin: 1rem 0;">{predicted_label}</h1>
    </div>
    ''', unsafe_allow_html=True)
    
    if prediction_proba is not None:
        # Create probability visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Probability bar chart
            fig_bar = px.bar(
                x=classes,
                y=prediction_proba,
                title="Prediction Probabilities",
                labels={'x': 'Damage Level', 'y': 'Probability'},
                color=prediction_proba,
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Probability pie chart
            fig_pie = px.pie(
                values=prediction_proba,
                names=classes,
                title="Probability Distribution"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Confidence metrics
        max_prob = np.max(prediction_proba)
        confidence_level = "High" if max_prob > 0.8 else "Medium" if max_prob > 0.6 else "Low"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence Level", confidence_level, f"{max_prob:.1%}")
        with col2:
            st.metric("Top Probability", f"{max_prob:.1%}")
        with col3:
            entropy = -np.sum(prediction_proba * np.log(prediction_proba + 1e-10))
            st.metric("Prediction Entropy", f"{entropy:.3f}")

def display_feature_importance(model_package, user_inputs):
    """Display feature importance and input analysis."""
    st.markdown('<div class="sub-header">Feature Analysis</div>', unsafe_allow_html=True)
    
    model = model_package['model']
    
    if hasattr(model, 'feature_importances_'):
        feature_names = model_package['feature_names']
        importances = model.feature_importances_
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances,
            'User_Value': [user_inputs.get(f, 0) for f in feature_names]
        }).sort_values('Importance', ascending=False)
        
        # Top 15 most important features
        top_features = importance_df.head(15)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Feature importance plot
            fig_importance = px.bar(
                top_features,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Top 15 Feature Importance",
                color='Importance',
                color_continuous_scale='blues'
            )
            fig_importance.update_layout(height=500)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
            st.markdown("**Your Input Values for Top Features:**")
            for _, row in top_features.head(10).iterrows():
                st.write(f"**{row['Feature']}**: {row['User_Value']:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)

def display_model_performance(summary):
    """Display model performance metrics."""
    st.markdown('<div class="sub-header">Model Performance</div>', unsafe_allow_html=True)
    
    if summary:
        perf_summary = summary.get('Model Performance Summary', {})
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Model Accuracy",
                f"{perf_summary.get('Final Accuracy', 0):.1%}",
                help="Overall prediction accuracy on test set"
            )
        
        with col2:
            st.metric(
                "F1 Score",
                f"{perf_summary.get('Final F1 Score', 0):.3f}",
                help="Weighted F1 score across all classes"
            )
        
        with col3:
            st.metric(
                "Features Used",
                perf_summary.get('Number of Features', 0),
                help="Total number of features in the model"
            )
        
        with col4:
            st.metric(
                "Classes",
                perf_summary.get('Number of Classes', 0),
                help="Number of damage categories predicted"
            )
        
        # Model comparison
        if 'All Models Comparison' in summary:
            st.markdown("**Model Comparison Results:**")
            
            models_data = []
            for model_name, metrics in summary['All Models Comparison'].items():
                models_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['Accuracy'],
                    'F1 Score': metrics['F1 Score'],
                    'CV Mean': metrics['CV Mean']
                })
            
            models_df = pd.DataFrame(models_data)
            
            fig_comparison = px.bar(
                models_df.melt(id_vars=['Model'], var_name='Metric', value_name='Score'),
                x='Model',
                y='Score',
                color='Metric',
                barmode='group',
                title="Model Performance Comparison"
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

def display_data_insights(sample_data):
    """Display insights about the training data."""
    st.markdown('<div class="sub-header">🔍 Training Data Insights</div>', unsafe_allow_html=True)
    
    if sample_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Dataset Statistics:**")
            st.write(f"• Total Records: {len(sample_data):,}")
            st.write(f"• Features: {len(sample_data.columns)}")
            st.write(f"• Missing Values: {sample_data.isnull().sum().sum()}")
            
            # Feature distribution
            numeric_features = sample_data.select_dtypes(include=[np.number]).columns[:5]
            if len(numeric_features) > 0:
                feature_to_plot = st.selectbox("Select Feature to Visualize:", numeric_features)
                
                fig_hist = px.histogram(
                    sample_data,
                    x=feature_to_plot,
                    title=f"Distribution of {feature_to_plot}",
                    marginal="box"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.markdown("**Feature Statistics:**")
            st.dataframe(sample_data.describe().round(4))

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">✈️ Aircraft Damage Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model and data
    model_package = load_model_package()
    if model_package is None:
        st.stop()
    
    sample_data = load_sample_data()
    summary = load_model_summary()
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode:",
        ["🔮 Make Prediction", "📊 Model Performance", "🔍 Data Insights", "ℹ️ About"]
    )
    
    if app_mode == "🔮 Make Prediction":
        st.markdown("### Enter aircraft and incident details to predict damage severity")
        
        # Feature input form
        user_inputs = create_feature_input_form(model_package, sample_data)
        
        # Prediction button
        if st.button("Predict Aircraft Damage", type="primary"):
            with st.spinner("Making prediction..."):
                predicted_label, prediction_proba, classes = make_prediction(model_package, user_inputs)
                
                # Display results
                display_prediction_results(predicted_label, prediction_proba, classes)
                
                # Feature importance analysis
                display_feature_importance(model_package, user_inputs)
                
                # Download prediction report
                if st.button("Download Prediction Report"):
                    report = {
                        'prediction': predicted_label,
                        'probabilities': dict(zip(classes, prediction_proba.tolist())) if prediction_proba is not None else {},
                        'input_features': user_inputs,
                        'model_type': model_package['model_type']
                    }
                    
                    st.download_button(
                        label="Download JSON Report",
                        data=json.dumps(report, indent=2),
                        file_name=f"prediction_report_{predicted_label}.json",
                        mime="application/json"
                    )
    
    elif app_mode == "Model Performance":
        display_model_performance(summary)
        
        # Additional performance visualizations
        if summary and 'Classification Report' in summary:
            st.markdown("### Detailed Classification Report")
            
            # Convert classification report to DataFrame for better display
            class_report = summary['Classification Report']
            
            # Extract per-class metrics
            class_metrics = []
            for class_name, metrics in class_report.items():
                if isinstance(metrics, dict) and class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    class_metrics.append({
                        'Class': class_name,
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1-score'],
                        'Support': metrics['support']
                    })
            
            if class_metrics:
                metrics_df = pd.DataFrame(class_metrics)
                
                # Metrics visualization
                fig_metrics = px.bar(
                    metrics_df.melt(id_vars=['Class'], 
                                   value_vars=['Precision', 'Recall', 'F1-Score'],
                                   var_name='Metric', value_name='Score'),
                    x='Class',
                    y='Score',
                    color='Metric',
                    barmode='group',
                    title="Per-Class Performance Metrics"
                )
                st.plotly_chart(fig_metrics, use_container_width=True)
                
                # Support visualization
                fig_support = px.bar(
                    metrics_df,
                    x='Class',
                    y='Support',
                    title="Sample Support per Class",
                    color='Support',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_support, use_container_width=True)
    
    elif app_mode == "Data Insights":
        display_data_insights(sample_data)
    
    elif app_mode == "About":
        st.markdown("### About This Application")
        st.markdown("""
        This **Aircraft Damage Prediction System** uses machine learning to predict the severity of aircraft damage
        based on various incident parameters.
        
        **🎯 Features:**
        - **Interactive Prediction**: Input aircraft and incident details to get damage predictions
        - **Probability Analysis**: View confidence levels and probability distributions
        - **Feature Importance**: Understand which factors most influence predictions
        - **Model Performance**: Review comprehensive model evaluation metrics
        - **Data Insights**: Explore the training dataset characteristics
        
        **🤖 Model Information:**
        """)
        
        if model_package:
            st.info(f"""
            - **Model Type**: {model_package['model_type']}
            - **Features**: {len(model_package['feature_names'])} input features
            - **Classes**: {len(model_package['label_encoder'].classes_)} damage categories
            - **Categories**: {', '.join(model_package['label_encoder'].classes_)}
            """)
        
        st.markdown("""
        **📊 Prediction Categories:**
        - **None**: No damage to aircraft
        - **Minor**: Light damage, aircraft repairable
        - **Substantial**: Significant damage affecting flight safety
        - **Destroyed**: Aircraft is beyond repair
        - **Unknown**: Damage level not determined
        
        **⚠️ Important Notes:**
        - This tool is for educational and research purposes
        - Predictions should not be used as the sole basis for operational decisions
        - Always consult aviation safety experts for critical assessments
        """)
        
        # Model training info
        with st.expander("Technical Details"):
            st.markdown("""
            **Data Processing:**
            - Comprehensive data cleaning and preprocessing
            - Feature engineering from temporal and categorical data
            - Label encoding for categorical variables
            - Standardization of numerical features
            
            **Model Training:**
            - Multiple algorithm comparison (Random Forest, Gradient Boosting, etc.)
            - Cross-validation for robust performance estimation
            - Hyperparameter optimization using GridSearch
            - Performance evaluation with multiple metrics
            
            **Model Deployment:**
            - Saved model with complete preprocessing pipeline
            - Interactive web interface using Streamlit
            - Real-time prediction with confidence analysis
            """)

if __name__ == "__main__":
    main()
