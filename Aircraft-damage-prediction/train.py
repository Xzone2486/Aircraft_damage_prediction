# aircraft_damage_prediction_model.py

"""
Aircraft Damage Prediction Model Training and Evaluation
Uses preprocessed aviation data to predict aircraft damage severity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, roc_auc_score, roc_curve, 
                           precision_recall_curve, f1_score)
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'preprocessed_data_file': 'AviationData_preprocessed.csv',
    'original_data_file': 'AviationData.csv',
    'model_save_path': 'aircraft_damage_model.pkl',
    'plots_directory': 'visualization_plots',
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5
}

# Create plots directory
Path(CONFIG['plots_directory']).mkdir(exist_ok=True)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data_for_modeling():
    """Load preprocessed data and original data for target extraction."""
    logger.info("Loading data for modeling...")
    
    try:
        # Load preprocessed features
        X_data = pd.read_csv(CONFIG['preprocessed_data_file'])
        logger.info(f"Preprocessed data loaded: {X_data.shape}")
        
        # Load original data to extract target variable
        original_data = pd.read_csv(CONFIG['original_data_file'], encoding='latin1')
        logger.info(f"Original data loaded: {original_data.shape}")
        
        return X_data, original_data
    
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_target_variable(original_data):
    """Create and encode target variable from original data."""
    logger.info("Creating target variable...")
    
    # Extract aircraft damage information
    if 'Aircraft_damage' in original_data.columns:
        # Remove duplicates and align with preprocessed data length
        original_data_clean = original_data.drop_duplicates().reset_index(drop=True)
        
        # Handle missing values in target
        target_raw = original_data_clean['Aircraft_damage'].fillna('Unknown')
        
        # Simplify damage categories for better modeling
        def simplify_damage(damage):
            if pd.isna(damage):
                return 'Unknown'
            damage_str = str(damage).lower()
            if 'destroyed' in damage_str:
                return 'Destroyed'
            elif 'substantial' in damage_str:
                return 'Substantial'
            elif 'minor' in damage_str:
                return 'Minor'
            elif 'none' in damage_str:
                return 'None'
            else:
                return 'Unknown'
        
        target_simplified = target_raw.apply(simplify_damage)
        
        # Encode target variable
        le_target = LabelEncoder()
        y_encoded = le_target.fit_transform(target_simplified)
        
        logger.info(f"Target variable created with classes: {le_target.classes_}")
        logger.info(f"Target distribution:\n{pd.Series(target_simplified).value_counts()}")
        
        return y_encoded, le_target, target_simplified
    
    else:
        raise ValueError("Aircraft_damage column not found in original data")

def perform_eda_and_visualization(X_data, y_data, target_names, save_plots=True):
    """Perform comprehensive EDA and create visualizations."""
    logger.info("Performing EDA and creating visualizations...")
    
    # 1. Dataset Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Dataset shape info
    axes[0, 0].text(0.1, 0.8, f'Dataset Shape: {X_data.shape}', fontsize=14, transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.1, 0.6, f'Features: {X_data.shape[1]}', fontsize=12, transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.1, 0.4, f'Samples: {X_data.shape[0]}', fontsize=12, transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.1, 0.2, f'Missing Values: {X_data.isnull().sum().sum()}', fontsize=12, transform=axes[0, 0].transAxes)
    axes[0, 0].set_title('Dataset Overview')
    axes[0, 0].axis('off')
    
    # Target distribution
    target_counts = pd.Series(target_names).value_counts()
    axes[0, 1].pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Aircraft Damage Distribution')
    
    # Feature correlation heatmap (top 20 features)
    corr_matrix = X_data.iloc[:, :20].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=axes[1, 0])
    axes[1, 0].set_title('Feature Correlation Matrix (Top 20 Features)')
    
    # Missing values per feature (if any)
    missing_data = X_data.isnull().sum().sort_values(ascending=False)[:20]
    if missing_data.sum() > 0:
        missing_data.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Missing Values by Feature (Top 20)')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Missing Values!', ha='center', va='center', 
                       transform=axes[1, 1].transAxes, fontsize=16, color='green')
        axes[1, 1].set_title('Missing Values Status')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{CONFIG['plots_directory']}/01_dataset_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Feature importance (using a simple Random Forest)
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=CONFIG['random_state'])
    rf_temp.fit(X_data, y_data)
    feature_importance = pd.DataFrame({
        'feature': X_data.columns,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=feature_importance, y='feature', x='importance', ax=axes[0, 0])
    axes[0, 0].set_title('Top 15 Feature Importance (Random Forest)')
    
    # Distribution of top features
    top_features = feature_importance.head(6)['feature'].tolist()
    for i, feature in enumerate(top_features[:4]):
        ax = axes[0, 1] if i < 2 else axes[1, 0] if i == 2 else axes[1, 1]
        if i >= 2:
            i -= 2
        
        X_data[feature].hist(bins=50, alpha=0.7, ax=ax)
        ax.set_title(f'Distribution: {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{CONFIG['plots_directory']}/02_feature_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Target vs Features Analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Box plots for top features vs target
    for i, feature in enumerate(top_features[:6]):
        df_temp = pd.DataFrame({
            'feature': X_data[feature],
            'target': target_names
        })
        sns.boxplot(data=df_temp, x='target', y='feature', ax=axes[i])
        axes[i].set_title(f'{feature} vs Aircraft Damage')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{CONFIG['plots_directory']}/03_target_vs_features.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return feature_importance

def train_multiple_models(X_train, X_test, y_train, y_test):
    """Train multiple machine learning models and compare performance."""
    logger.info("Training multiple models...")
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=CONFIG['random_state'],
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=CONFIG['random_state']
        ),
        'Logistic Regression': LogisticRegression(
            random_state=CONFIG['random_state'],
            max_iter=1000,
            multi_class='ovr'
        ),
        'SVM': SVC(
            kernel='rbf',
            random_state=CONFIG['random_state'],
            probability=True
        )
    }
    
    # Train and evaluate models
    model_results = {}
    trained_models = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=CONFIG['cv_folds'], scoring='accuracy')
        
        model_results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    
    return model_results, trained_models

def visualize_model_comparison(model_results, le_target, y_test, save_plots=True):
    """Create comprehensive model comparison visualizations."""
    logger.info("Creating model comparison visualizations...")
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    model_names = list(model_results.keys())
    accuracies = [model_results[name]['accuracy'] for name in model_names]
    f1_scores = [model_results[name]['f1_score'] for name in model_names]
    cv_means = [model_results[name]['cv_mean'] for name in model_names]
    cv_stds = [model_results[name]['cv_std'] for name in model_names]
    
    x_pos = np.arange(len(model_names))
    
    axes[0, 0].bar(x_pos, accuracies, alpha=0.7, color='skyblue', label='Test Accuracy')
    axes[0, 0].bar(x_pos, cv_means, alpha=0.7, color='lightcoral', label='CV Mean')
    axes[0, 0].errorbar(x_pos, cv_means, yerr=cv_stds, fmt='none', color='red', capsize=5)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score comparison
    axes[0, 1].bar(x_pos, f1_scores, alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].set_title('Model F1 Score Comparison')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(model_names, rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Best model confusion matrix - FIXED
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
    best_result = model_results[best_model_name]
    
    # FIXED: Use y_test instead of model.classes_
    cm = confusion_matrix(y_test, best_result['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_target.classes_, 
                yticklabels=le_target.classes_, ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Performance metrics comparison
    metrics_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': accuracies,
        'F1_Score': f1_scores,
        'CV_Mean': cv_means
    })
    
    # Simple bar chart instead of radar (easier to implement)
    metrics_df.set_index('Model')[['Accuracy', 'F1_Score', 'CV_Mean']].plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_title('Model Metrics Comparison')
    axes[1, 1].set_xlabel('Models')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{CONFIG['plots_directory']}/04_model_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model_name

def visualize_best_model_details(best_model_name, model_results, X_test, y_test, le_target, save_plots=True):
    """Create detailed visualizations for the best performing model."""
    logger.info(f"Creating detailed visualizations for {best_model_name}...")
    
    best_result = model_results[best_model_name]
    model = best_result['model']
    y_pred = best_result['y_pred']
    y_pred_proba = best_result['y_pred_proba']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Detailed Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le_target.classes_, 
                yticklabels=le_target.classes_, ax=axes[0, 0])
    axes[0, 0].set_title(f'Detailed Confusion Matrix - {best_model_name}')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # 2. Feature Importance (if available)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': X_test.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        sns.barplot(data=feature_importance, y='feature', x='importance', ax=axes[0, 1])
        axes[0, 1].set_title(f'Feature Importance - {best_model_name}')
    else:
        axes[0, 1].text(0.5, 0.5, 'Feature Importance\nNot Available\nfor this model', 
                       ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=14)
        axes[0, 1].set_title('Feature Importance')
    
    # 3. ROC Curves (for multiclass)
    if y_pred_proba is not None and len(le_target.classes_) > 2:
        # Plot ROC curve for each class
        for i, class_name in enumerate(le_target.classes_):
            y_test_binary = (y_test == i).astype(int)
            if len(np.unique(y_test_binary)) > 1:  # Only plot if both classes present
                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba[:, i])
                auc_score = roc_auc_score(y_test_binary, y_pred_proba[:, i])
                axes[1, 0].plot(fpr, tpr, label=f'{class_name} (AUC = {auc_score:.3f})')
        
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curves - Multiclass')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'ROC Curve\nNot Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
        axes[1, 0].set_title('ROC Curve')
    
    # 4. Prediction Distribution
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    true_counts = pd.Series(y_test).value_counts().sort_index()
    
    x_pos = np.arange(len(le_target.classes_))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, [pred_counts.get(i, 0) for i in range(len(le_target.classes_))], 
                   width, label='Predicted', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, [true_counts.get(i, 0) for i in range(len(le_target.classes_))], 
                   width, label='Actual', alpha=0.7)
    
    axes[1, 1].set_xlabel('Damage Classes')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction vs Actual Distribution')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(le_target.classes_, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(f"{CONFIG['plots_directory']}/05_best_model_details.png", dpi=300, bbox_inches='tight')
    plt.show()

def optimize_best_model(best_model_name, model_results, X_train, y_train):
    """Optimize the best model using GridSearch."""
    logger.info(f"Optimizing {best_model_name} with GridSearch...")
    
    best_model = model_results[best_model_name]['model']
    
    # Define parameter grids for different models
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        },
        'Gradient Boosting': {
            'n_estimators': [100, 200],
            'max_depth': [6, 8],
            'learning_rate': [0.1, 0.05],
            'min_samples_split': [2, 5]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000, 2000]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
    
    if best_model_name in param_grids:
        param_grid = param_grids[best_model_name]
        
        # Create a new instance of the best model
        if best_model_name == 'Random Forest':
            model = RandomForestClassifier(random_state=CONFIG['random_state'], n_jobs=-1)
        elif best_model_name == 'Gradient Boosting':
            model = GradientBoostingClassifier(random_state=CONFIG['random_state'])
        elif best_model_name == 'Logistic Regression':
            model = LogisticRegression(random_state=CONFIG['random_state'], multi_class='ovr')
        elif best_model_name == 'SVM':
            model = SVC(random_state=CONFIG['random_state'], probability=True)
        
        # Perform GridSearch
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    else:
        logger.info("Using original best model (no optimization parameters defined)")
        return best_model

def save_model_and_components(final_model, le_target, feature_names):
    """Save the final model and related components."""
    logger.info("Saving model and components...")
    
    # Create model package
    model_package = {
        'model': final_model,
        'label_encoder': le_target,
        'feature_names': feature_names,
        'model_type': type(final_model).__name__,
        'config': CONFIG
    }
    
    # Save using joblib (better for sklearn models)
    joblib.dump(model_package, CONFIG['model_save_path'])
    logger.info(f"Model saved to {CONFIG['model_save_path']}")
    
    # Also save as pickle backup
    with open(CONFIG['model_save_path'].replace('.pkl', '_backup.pkl'), 'wb') as f:
        pickle.dump(model_package, f)
    
    return model_package

def create_model_summary_report(model_results, best_model_name, final_model, X_test, y_test, le_target):
    """Create a comprehensive model summary report."""
    logger.info("Creating model summary report...")
    
    # Generate classification report for the final model
    y_pred_final = final_model.predict(X_test)
    class_report = classification_report(y_test, y_pred_final, 
                                       target_names=le_target.classes_, 
                                       output_dict=True)
    
    # Create summary
    summary = {
        'Model Performance Summary': {
            'Best Model': best_model_name,
            'Final Accuracy': accuracy_score(y_test, y_pred_final),
            'Final F1 Score': f1_score(y_test, y_pred_final, average='weighted'),
            'Test Set Size': len(y_test),
            'Number of Features': len(X_test.columns),
            'Number of Classes': len(le_target.classes_)
        },
        'All Models Comparison': {
            name: {
                'Accuracy': results['accuracy'],
                'F1 Score': results['f1_score'],
                'CV Mean': results['cv_mean'],
                'CV Std': results['cv_std']
            }
            for name, results in model_results.items()
        },
        'Classification Report': class_report
    }
    
    # Save summary to file
    import json
    with open(f"{CONFIG['plots_directory']}/model_summary_report.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*50)
    print("AIRCRAFT DAMAGE PREDICTION MODEL SUMMARY")
    print("="*50)
    print(f"Best Model: {best_model_name}")
    print(f"Final Accuracy: {summary['Model Performance Summary']['Final Accuracy']:.4f}")
    print(f"Final F1 Score: {summary['Model Performance Summary']['Final F1 Score']:.4f}")
    print(f"Number of Features: {summary['Model Performance Summary']['Number of Features']}")
    print(f"Number of Classes: {summary['Model Performance Summary']['Number of Classes']}")
    print(f"Classes: {', '.join(le_target.classes_)}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_final, target_names=le_target.classes_))
    print("="*50)
    
    return summary

def main():
    """Main pipeline for aircraft damage prediction model training."""
    logger.info("Starting Aircraft Damage Prediction Model Training Pipeline...")
    
    # 1. Load data
    X_data, original_data = load_data_for_modeling()
    
    # 2. Create target variable
    y_data, le_target, target_names = create_target_variable(original_data)
    
    # Ensure X and y have the same length
    min_length = min(len(X_data), len(y_data))
    X_data = X_data.iloc[:min_length]
    y_data = y_data[:min_length]
    target_names = target_names[:min_length]
    
    logger.info(f"Final dataset shape: X{X_data.shape}, y{y_data.shape}")
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, 
        test_size=CONFIG['test_size'], 
        random_state=CONFIG['random_state'],
        stratify=y_data
    )
    
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # 4. Perform EDA and visualization
    feature_importance = perform_eda_and_visualization(X_data, y_data, target_names)
    
    # 5. Train multiple models
    model_results, trained_models = train_multiple_models(X_train, X_test, y_train, y_test)
    
    # 6. Visualize model comparison - FIXED: Added y_test parameter
    best_model_name = visualize_model_comparison(model_results, le_target, y_test)
    
    # 7. Detailed visualization of best model
    visualize_best_model_details(best_model_name, model_results, X_test, y_test, le_target)
    
    # 8. Optimize best model
    final_model = optimize_best_model(best_model_name, model_results, X_train, y_train)
    
    # 9. Save model and components
    model_package = save_model_and_components(final_model, le_target, X_data.columns.tolist())
    
    # 10. Create final summary report
    summary = create_model_summary_report(model_results, best_model_name, final_model, 
                                        X_test, y_test, le_target)
    
    logger.info("Model training pipeline completed successfully!")
    
    return final_model, model_package, summary

# Example usage and model prediction function
def predict_aircraft_damage(model_package, new_data):
    """
    Function to make predictions on new data using the saved model.
    
    Args:
        model_package: Loaded model package
        new_data: DataFrame with preprocessed features
    
    Returns:
        predictions and probabilities
    """
    model = model_package['model']
    le_target = model_package['label_encoder']
    
    # Make predictions
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data) if hasattr(model, 'predict_proba') else None
    
    # Convert predictions back to original labels
    predicted_labels = le_target.inverse_transform(predictions)
    
    return predicted_labels, probabilities

if __name__ == "__main__":
    # Run the complete pipeline
    final_model, model_package, summary = main()
    
    print("\n🎉 Aircraft Damage Prediction Model Training Complete!")
    print(f"📁 Model saved to: {CONFIG['model_save_path']}")
    print(f"📊 Visualizations saved to: {CONFIG['plots_directory']}/")
    print(f"📋 Summary report saved to: {CONFIG['plots_directory']}/model_summary_report.json")
    
    # Example of how to load and use the model
    print("\n" + "="*50)
    print("EXAMPLE: Loading and Using the Saved Model")
    print("="*50)
    
    # Load the saved model
    loaded_model_package = joblib.load(CONFIG['model_save_path'])
    print(f"✅ Model loaded successfully!")
    print(f"Model type: {loaded_model_package['model_type']}")
    print(f"Features expected: {len(loaded_model_package['feature_names'])}")
    print(f"Classes: {loaded_model_package['label_encoder'].classes_}")  # FIXED: Removed extra parenthesis
