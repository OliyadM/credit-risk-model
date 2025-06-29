import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = "data/raw/data.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
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

            # For categorical columns, use mode (most frequent)
            ProviderId_mode=('ProviderId', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            ProductCategory_mode=('ProductCategory', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            ChannelId_mode=('ChannelId', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
            PricingStrategy_mode=('PricingStrategy', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan),
        ).fillna(0).reset_index()

        return agg

def load_data(path=DATA_PATH):
    return pd.read_csv(path)

def build_pipeline():
    numerical_features = [
        "total_amount", "avg_amount", "num_transactions",
        "amount_std", "num_unique_products", "num_channels_used",
        "avg_transaction_hour", "avg_transaction_day",
        "avg_transaction_month", "avg_transaction_year",
        # Note PricingStrategy_mode may be int or categorical; treat as categorical below
    ]

    categorical_features = [
        "ProviderId_mode", "ProductCategory_mode", "ChannelId_mode", "PricingStrategy_mode"
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    full_pipeline = Pipeline([
        ("date_features", DateFeatureExtractor()),
        ("aggregator", CustomerAggregator()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline

def process_and_save(input_path=DATA_PATH, output_path=PROCESSED_PATH):
    df = load_data(input_path)
    pipeline = build_pipeline()
    processed = pipeline.fit_transform(df)

    num_cols = [
        "total_amount", "avg_amount", "num_transactions",
        "amount_std", "num_unique_products", "num_channels_used",
        "avg_transaction_hour", "avg_transaction_day",
        "avg_transaction_month", "avg_transaction_year",
    ]

    cat_pipeline = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_cols = cat_pipeline.named_steps["encoder"].get_feature_names_out()

    all_cols = num_cols + list(cat_cols)
    processed_df = pd.DataFrame(processed, columns=all_cols)

    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    process_and_save()
