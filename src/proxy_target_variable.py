import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# File paths
PROCESSED_PATH = "data/processed/data_cleaned.csv"
FEATURED_PATH = "data/processed/processed_data.csv"

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        # Ensure 'TransactionStartTime' is a datetime object and remove timezone
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')
        df['TransactionStartTime'] = df['TransactionStartTime'].dt.tz_localize(None)

        # Extract features from the transaction datetime
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        return df

class CustomerAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        
        # Retain the 'TransactionStartTime' column for RFM calculation
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

        # Aggregating customer-level features based on their 'CustomerId'
        agg = df.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            num_transactions=('TransactionId', 'count'),
            amount_std=('Amount', 'std'),
            num_unique_products=('ProductId', 'nunique'),
            num_channels_used=('ChannelId', 'nunique'),
            avg_transaction_hour=('transaction_hour', 'mean'),
            avg_transaction_day=('transaction_day', 'mean'),
            avg_transaction_month=('transaction_month', 'mean'),
            avg_transaction_year=('transaction_year', 'mean'),
            
            # Categorical features: Mode (most frequent) value
            ProviderId_mode=('ProviderId', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            ProductCategory_mode=('ProductCategory', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            ChannelId_mode=('ChannelId', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            PricingStrategy_mode=('PricingStrategy', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            
            # Keep the first transaction time per customer
            first_transaction_time=('TransactionStartTime', 'min')
        ).fillna(0).reset_index()  # Filling missing values with 0

        return agg

def load_data(path=PROCESSED_PATH):
    """Load the cleaned data"""
    return pd.read_csv(path)

def build_feature_engineering_pipeline():
    """Build and return the feature engineering pipeline"""
    numerical_features = [
        "total_amount", "avg_amount", "num_transactions",
        "amount_std", "num_unique_products", "num_channels_used",
        "avg_transaction_hour", "avg_transaction_day",
        "avg_transaction_month", "avg_transaction_year",
    ]

    categorical_features = [
        "ProviderId_mode", "ProductCategory_mode", "ChannelId_mode", "PricingStrategy_mode"
    ]

    # Pipeline for numerical features
    numeric_pipeline = Pipeline([ 
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Pipeline for categorical features
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combining the transformations for each feature type
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # Complete feature engineering pipeline
    feature_engineering_pipeline = Pipeline([
        ("date_features", DateFeatureExtractor()),
        ("aggregator", CustomerAggregator()),
        ("preprocessor", preprocessor)
    ])

    return feature_engineering_pipeline

def process_feature_engineering(input_path=PROCESSED_PATH, output_path=FEATURED_PATH):
    """Process the feature engineering and save the transformed data"""
    df = load_data(input_path)
    feature_pipeline = build_feature_engineering_pipeline()

    # Apply transformations
    processed_data = feature_pipeline.fit_transform(df)

    # Extract column names for the transformed features
    num_cols = [
        "total_amount", "avg_amount", "num_transactions",
        "amount_std", "num_unique_products", "num_channels_used",
        "avg_transaction_hour", "avg_transaction_day",
        "avg_transaction_month", "avg_transaction_year",
    ]

    cat_pipeline = feature_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_cols = cat_pipeline.named_steps["encoder"].get_feature_names_out()

    all_cols = num_cols + list(cat_cols)

    # Create a DataFrame with the transformed features
    processed_df = pd.DataFrame(processed_data, columns=all_cols)

    # Save the processed data to the specified output path
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    process_feature_engineering()

