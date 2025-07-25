# aviation_data_preprocessing_optimized.py

"""
Optimized Python script for aviation data preprocessing.
Performs data cleaning, transformation, and feature engineering
for machine learning model training with memory optimization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
import logging
from typing import List, Dict, Tuple
import gc

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'input_file': 'AviationData.csv',
    'output_file': 'AviationData_preprocessed.csv',
    'encoding': 'latin1',
    'top_n_categories': 20,  # Reduced from 50 to save memory
    'random_state': 42,
    'use_label_encoding': True  # Use label encoding instead of one-hot for high cardinality
}

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def load_data(file_path: str, encoding: str = 'latin1') -> pd.DataFrame:
    """Load aviation data with robust error handling."""
    try:
        path = Path(file_path)
        if not path.exists():
            path = Path.cwd() / file_path
            
        df = pd.read_csv(path, encoding=encoding, low_memory=False)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def initial_data_exploration(df: pd.DataFrame) -> Dict:
    """Perform initial data exploration and quality checks."""
    logger.info("Performing initial data exploration...")
    
    # Remove duplicates efficiently
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - len(df)
    
    # Calculate missing values
    missing_info = pd.DataFrame({
        'Missing_Count': df.isnull().sum(),
        'Missing_Percentage': (df.isnull().sum() / len(df)) * 100
    }).sort_values('Missing_Percentage', ascending=False)
    
    results = {
        'initial_shape': (initial_rows, df.shape[1]),
        'final_shape': df.shape,
        'duplicates_removed': duplicates_removed,
        'missing_info': missing_info[missing_info['Missing_Count'] > 0]
    }
    
    logger.info(f"Duplicates removed: {duplicates_removed}")
    logger.info(f"Columns with missing values: {len(results['missing_info'])}")
    
    return results


def convert_coordinates_to_numeric(series: pd.Series) -> pd.Series:
    """Convert coordinate strings to numeric values."""
    def parse_coordinate(coord_str):
        if pd.isna(coord_str) or coord_str == '':
            return np.nan
        
        # Try to convert directly to float first
        try:
            return float(coord_str)
        except (ValueError, TypeError):
            pass
        
        # Handle coordinate formats like '341525N', '1181034W'
        coord_str = str(coord_str).strip().upper()
        
        # Check if it's a coordinate format (ends with N, S, E, W)
        if coord_str and coord_str[-1] in 'NSEW':
            direction = coord_str[-1]
            coord_digits = coord_str[:-1]
            
            try:
                # Parse degrees, minutes, seconds format
                if len(coord_digits) >= 6:
                    if len(coord_digits) == 6:  # DDMMSS
                        degrees = int(coord_digits[:2])
                        minutes = int(coord_digits[2:4])
                        seconds = int(coord_digits[4:6])
                    elif len(coord_digits) == 7:  # DDDMMSS
                        degrees = int(coord_digits[:3])
                        minutes = int(coord_digits[3:5])
                        seconds = int(coord_digits[5:7])
                    else:
                        return np.nan
                    
                    # Convert to decimal degrees
                    decimal_degrees = degrees + minutes/60 + seconds/3600
                    
                    # Apply direction (S and W are negative)
                    if direction in 'SW':
                        decimal_degrees = -decimal_degrees
                    
                    return decimal_degrees
                else:
                    return np.nan
            except (ValueError, IndexError):
                return np.nan
        
        return np.nan
    
    return series.apply(parse_coordinate)


def process_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Process date columns and create time-based features."""
    logger.info("Processing date columns...")
    
    date_columns = ['Event_Date', 'Publication.Date']
    
    # Convert to datetime efficiently with proper format handling
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
    
    # Remove rows where Event_Date is missing (critical field)
    initial_rows = len(df)
    df.dropna(subset=['Event_Date'], inplace=True)
    logger.info(f"Rows removed due to missing Event_Date: {initial_rows - len(df)}")
    
    # Feature engineering from Event_Date
    if 'Event_Date' in df.columns:
        df['Event_Year'] = df['Event_Date'].dt.year
        df['Event_Month'] = df['Event_Date'].dt.month
        df['Event_Day'] = df['Event_Date'].dt.day
        df['Event_DayOfWeek'] = df['Event_Date'].dt.dayofweek
        df['Event_Quarter'] = df['Event_Date'].dt.quarter
        
        # Calculate report lag if both dates exist
        if 'Publication.Date' in df.columns:
            df['Report_Lag_Days'] = (df['Publication.Date'] - df['Event_Date']).dt.days
            median_lag = df['Report_Lag_Days'].median()
            df['Report_Lag_Days'] = df['Report_Lag_Days'].fillna(median_lag)
            df['Report_Lag_Days'] = df['Report_Lag_Days'].clip(0, 3650)  # Max 10 years
    
    return df


def process_injury_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Process injury-related columns and create derived features."""
    logger.info("Processing injury columns...")
    
    injury_cols = ['Total_Fatal_Injuries', 'Total_Serious_Injuries', 
                   'Total_Minor_Injuries', 'Total_Uninjured']
    
    # Ensure injury columns exist and are numeric
    for col in injury_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Create derived features
    if all(col in df.columns for col in injury_cols):
        df['Total_Injuries'] = (df['Total_Fatal_Injuries'] + 
                               df['Total_Serious_Injuries'] + 
                               df['Total_Minor_Injuries'])
        df['Total_Occupants'] = df['Total_Injuries'] + df['Total_Uninjured']
        
        # Create injury severity flags
        df['Has_Fatal'] = (df['Total_Fatal_Injuries'] > 0).astype(int)
        df['Has_Serious'] = (df['Total_Serious_Injuries'] > 0).astype(int)
        df['Has_Minor'] = (df['Total_Minor_Injuries'] > 0).astype(int)
    
    # Process Injury_Severity column
    if 'Injury_Severity' in df.columns:
        df['Simplified_Injury_Severity'] = df['Injury_Severity'].apply(_simplify_severity)
    
    return df


def _simplify_severity(severity) -> str:
    """Helper function to simplify injury severity categories."""
    if pd.isna(severity):
        return 'Unknown'
    
    severity_str = str(severity).lower()
    if 'fatal' in severity_str:
        return 'Fatal'
    elif 'serious' in severity_str:
        return 'Serious'
    elif 'minor' in severity_str:
        return 'Minor'
    elif 'non-fatal' in severity_str:
        return 'Non-Fatal'
    else:
        return 'Unknown'


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values using appropriate strategies for different column types."""
    logger.info("Handling missing values...")
    
    # Define imputation strategies
    strategies = {
        'fill_unknown': [
            'Aircraft_damage', 'Aircraft_Category', 'Make', 'Model', 'Engine_Type',
            'FAR_Description', 'Schedule', 'Purpose_of_flight', 'Air_carrier',
            'Weather_Condition', 'Broad_phase_of_flight', 'Report_Status'
        ],
        'fill_mode': ['Amateur_Built', 'Number_of_Engines', 'Country'],
        'fill_median': ['Latitude', 'Longitude']
    }
    
    # Apply imputation strategies
    for strategy, columns in strategies.items():
        existing_cols = [col for col in columns if col in df.columns]
        
        if strategy == 'fill_unknown':
            for col in existing_cols:
                df[col] = df[col].fillna('Unknown')
                
        elif strategy == 'fill_mode':
            for col in existing_cols:
                if df[col].isnull().any():
                    mode_value = df[col].mode()
                    if not mode_value.empty:
                        df[col] = df[col].fillna(mode_value.iloc[0])
                        
        elif strategy == 'fill_median':
            for col in existing_cols:
                if df[col].isnull().any():
                    # Convert to numeric first, handling coordinate formats
                    if col in ['Latitude', 'Longitude']:
                        df[col] = convert_coordinates_to_numeric(df[col])
                    else:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Fill with median only if we have numeric data
                    if df[col].dtype in ['int64', 'float64']:
                        df[col] = df[col].fillna(df[col].median())
                    else:
                        # If conversion failed, treat as categorical
                        df[col] = df[col].fillna('Unknown')
    
    # Handle sparse columns (Airport info)
    if 'Airport_Code' in df.columns:
        df['Airport_Related'] = df['Airport_Code'].notna().astype(int)
        df.drop(columns=['Airport_Code', 'Airport_Name'], inplace=True, errors='ignore')
    
    # Drop high-cardinality, low-value columns
    cols_to_drop = ['Location', 'Event_Id', 'Accident_Number', 'Registration_Number']
    df.drop(columns=[col for col in cols_to_drop if col in df.columns], 
            inplace=True, errors='ignore')
    
    return df


def convert_feature_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert features to appropriate data types."""
    logger.info("Converting feature types...")
    
    # Convert Amateur_Built to binary
    if 'Amateur_Built' in df.columns:
        df['Amateur_Built'] = df['Amateur_Built'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    
    # Convert Number_of_Engines to numeric
    if 'Number_of_Engines' in df.columns:
        df['Number_of_Engines'] = pd.to_numeric(df['Number_of_Engines'], errors='coerce')
        mode_engines = df['Number_of_Engines'].mode()
        if not mode_engines.empty:
            df['Number_of_Engines'] = df['Number_of_Engines'].fillna(mode_engines.iloc[0]).astype(int)
    
    return df


def reduce_cardinality_aggressive(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Aggressively reduce cardinality of high-cardinality categorical features."""
    logger.info(f"Aggressively reducing cardinality for memory optimization (top {top_n})...")
    
    high_cardinality_cols = ['Make', 'Model']
    
    for col in high_cardinality_cols:
        if col in df.columns:
            # Get top N categories
            top_categories = df[col].value_counts().head(top_n).index.tolist()
            # Replace others with 'Other'
            df[col] = df[col].where(df[col].isin(top_categories), 'Other')
            logger.info(f"Reduced {col} cardinality to {df[col].nunique()} categories")
    
    return df


def apply_label_encoding(df: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Apply label encoding to categorical features to save memory."""
    logger.info("Applying label encoding to categorical features...")
    
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_features:
        if col in df_encoded.columns:
            le = LabelEncoder()
            # Handle any remaining NaN values
            df_encoded[col] = df_encoded[col].fillna('Unknown')
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Label encoded {col}: {len(le.classes_)} unique values")
    
    return df_encoded, label_encoders


def prepare_features_memory_efficient(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify and prepare features for encoding and scaling with memory efficiency."""
    # Identify feature types
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove date columns from numerical features
    date_columns = ['Event_Date', 'Publication.Date']
    numerical_features = [f for f in numerical_features if f not in date_columns]
    
    # Remove target variable from features if it exists
    target_candidates = ['Simplified_Injury_Severity', 'Injury_Severity']
    categorical_features = [f for f in categorical_features if f not in target_candidates]
    
    # Further reduce categorical features by removing very high cardinality ones
    # or ones with too many missing values
    categorical_features_filtered = []
    for col in categorical_features:
        if col in df.columns:
            unique_count = df[col].nunique()
            missing_pct = df[col].isnull().sum() / len(df) * 100
            
            # Keep only categorical features with reasonable cardinality and not too many missing values
            if unique_count <= 100 and missing_pct < 80:  # Adjust thresholds as needed
                categorical_features_filtered.append(col)
            else:
                logger.info(f"Dropping {col} - too many unique values ({unique_count}) or missing values ({missing_pct:.1f}%)")
    
    categorical_features = categorical_features_filtered
    
    logger.info(f"Selected {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
    
    return numerical_features, categorical_features


def create_memory_efficient_pipeline(numerical_features: List[str]) -> StandardScaler:
    """Create a memory-efficient preprocessing pipeline."""
    return StandardScaler()


def main():
    """Main preprocessing pipeline with memory optimization."""
    logger.info("Starting aviation data preprocessing with memory optimization...")
    
    # Load data
    df = load_data(CONFIG['input_file'], CONFIG['encoding'])
    
    # Initial exploration
    exploration_results = initial_data_exploration(df)
    
    # Data processing steps
    df = process_date_columns(df)
    df = process_injury_columns(df)
    df = handle_missing_values(df)
    df = convert_feature_types(df)
    df = reduce_cardinality_aggressive(df, CONFIG['top_n_categories'])
    
    # Force garbage collection
    gc.collect()
    
    # Prepare features for ML with memory efficiency
    numerical_features, categorical_features = prepare_features_memory_efficient(df)
    
    # Apply label encoding instead of one-hot encoding to save memory
    if categorical_features:
        df_encoded, label_encoders = apply_label_encoding(df, categorical_features)
        
        # Update numerical features to include the label encoded categorical features
        numerical_features.extend(categorical_features)
    else:
        df_encoded = df.copy()
        label_encoders = {}
    
    # Create feature matrix with only numerical features (including label encoded ones)
    feature_columns = [col for col in numerical_features if col in df_encoded.columns]
    X = df_encoded[feature_columns].copy()
    
    # Fill any remaining NaN values
    X = X.fillna(0)
    
    # Apply scaling
    scaler = StandardScaler()
    logger.info("Applying scaling to numerical features...")
    X_scaled = scaler.fit_transform(X)
    
    # Create final DataFrame
    df_processed = pd.DataFrame(X_scaled, columns=feature_columns)
    
    # Final validation
    logger.info(f"Final processed data shape: {df_processed.shape}")
    logger.info(f"Missing values in processed data: {df_processed.isnull().sum().sum()}")
    logger.info(f"Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Save processed data
    output_path = Path(CONFIG['output_file'])
    df_processed.to_csv(output_path, index=False)
    logger.info(f"Processed data saved to {output_path}")
    
    # Save label encoders for future use
    if label_encoders:
        import pickle
        encoders_path = Path('label_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(label_encoders, f)
        logger.info(f"Label encoders saved to {encoders_path}")
    
    # Save scaler for future use
    import pickle
    scaler_path = Path('scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    
    return df, df_processed, scaler, label_encoders


if __name__ == "__main__":
    original_df, processed_df, scaler, encoders = main()
    print("\nPreprocessing complete! Data is ready for machine learning.")
    print(f"Final dataset shape: {processed_df.shape}")
    print(f"Features available: {list(processed_df.columns)}")
