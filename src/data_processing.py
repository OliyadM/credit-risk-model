# src/data_processing.py
# This script handles feature engineering (Task 3) and proxy target variable engineering (Task 4).
# It uses sklearn pipelines for transformation steps where possible.
# Assumptions:
# - Raw data file is named 'data.csv' in ../data/raw/ (change if different, e.g., 'Train.csv').
# - Custom WOE implemented due to compatibility issues with xverse and recent pandas versions.
# - Binning uses pd.qcut with 5 bins for numerical features; no monotonic binning for simplicity.
# - ProductId is included in one-hot encoding despite having ~23 unique values, as it's manageable.
# - CountryCode and CurrencyCode are constants, so ignored in aggregation.
# - No missing values based on EDA, but fillna(0) applied for any std NaNs (e.g., single transaction customers).
# - To suppress dtype warnings in scaler, cast numerical columns to float after aggregation.
# - Added handling for constant/low-variance columns to prevent qcut failures and type errors.

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['TransactionStartTime'] = pd.to_datetime(X_copy['TransactionStartTime'])
        X_copy['Hour'] = X_copy['TransactionStartTime'].dt.hour
        X_copy['Day'] = X_copy['TransactionStartTime'].dt.day
        X_copy['Month'] = X_copy['TransactionStartTime'].dt.month
        X_copy['WeekDay'] = X_copy['TransactionStartTime'].dt.weekday
        X_copy['DayOfYear'] = X_copy['TransactionStartTime'].dt.dayofyear
        return X_copy

class RFMTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        snapshot_date = X_copy['TransactionStartTime'].max() + pd.Timedelta(days=1)
        rfm = X_copy.groupby('CustomerId').agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
            'TransactionId': 'count',
            'Value': 'sum'
        }).reset_index()
        rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']
        return rfm

class ClusterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, X, y=None):
        scaled_features = self.scaler.fit_transform(X[['Recency', 'Frequency', 'Monetary']])
        self.kmeans.fit(scaled_features)
        return self

    def transform(self, X):
        scaled_features = self.scaler.transform(X[['Recency', 'Frequency', 'Monetary']])
        X['Cluster'] = self.kmeans.predict(scaled_features)
        cluster_means = X.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
        cluster_means['RiskScore'] = cluster_means['Recency'] - cluster_means['Frequency'] - cluster_means['Monetary']
        high_risk_cluster = cluster_means['RiskScore'].idxmax()
        X['is_high_risk'] = (X['Cluster'] == high_risk_cluster).astype(int)
        return X

class FeatureAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        cat_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId', 'PricingStrategy']
        X_copy = pd.get_dummies(X_copy, columns=cat_cols)
        agg_dict = {
            'Amount': ['sum', 'mean', 'std', 'min', 'max'],
            'Value': ['sum', 'mean', 'std', 'min', 'max'],
            'Hour': ['mean', 'std'],
            'Day': ['mean', 'std'],
            'Month': ['mean', 'std'],
            'WeekDay': ['mean', 'std'],
            'DayOfYear': ['mean', 'std'],
            'FraudResult': ['sum', 'mean']
        }
        agg_dict_full = {f'{col}_{func}': (col, func) for col, funcs in agg_dict.items() for func in funcs}
        dummy_cols = [col for col in X_copy.columns if any(col.startswith(c + '_') for c in cat_cols)]
        for dcol in dummy_cols:
            agg_dict_full[dcol] = (dcol, 'sum')
        agg_dict_full['Transaction_Count'] = ('TransactionId', 'count')
        aggregated = X_copy.groupby('CustomerId').agg(**agg_dict_full).reset_index()
        aggregated.fillna(0, inplace=True)
        # Cast to float to avoid dtype warnings in scaler
        for col in aggregated.columns[1:]:
            aggregated[col] = aggregated[col].astype(float)
        return aggregated

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X.iloc[:, 1:])
        return self

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled.iloc[:, 1:] = self.scaler.transform(X.iloc[:, 1:])
        return X_scaled

class CustomWOE(BaseEstimator, TransformerMixin):
    def __init__(self, bins=5):
        self.bins = bins
        self.woe_dict = {}
        self.iv_dict = {}

    def fit(self, X, y):
        for col in X.columns:
            if X[col].nunique() <= 1:
                self.iv_dict[col] = 0.0
                self.woe_dict[col] = 'constant'
                continue

            df_temp = pd.DataFrame({'X': X[col], 'y': y})
            try:
                df_temp['bin'] = pd.qcut(df_temp['X'], q=self.bins, duplicates='drop')
            except ValueError:
                # Fallback for near-constant or unbinable columns
                df_temp['bin'] = pd.cut(df_temp['X'], bins=min(self.bins, X[col].nunique()), duplicates='drop')

            if df_temp['bin'].nunique() <= 1:
                self.iv_dict[col] = 0.0
                self.woe_dict[col] = 'constant'
                continue

            bin_grouped = df_temp.groupby('bin', observed=False)['y'].agg(['count', 'sum'])
            bin_grouped['good'] = bin_grouped['count'] - bin_grouped['sum']  # assuming 0=good, 1=bad
            bin_grouped['bad'] = bin_grouped['sum']
            bin_grouped['dist_good'] = bin_grouped['good'] / (bin_grouped['good'].sum() + 1e-10)
            bin_grouped['dist_bad'] = bin_grouped['bad'] / (bin_grouped['bad'].sum() + 1e-10)
            bin_grouped['woe'] = np.log(bin_grouped['dist_good'] / bin_grouped['dist_bad'] + 1e-10)
            bin_grouped['woe'] = bin_grouped['woe'].replace([np.inf, -np.inf], 0)

            self.woe_dict[col] = dict(zip(bin_grouped.index, bin_grouped['woe']))

            iv = ((bin_grouped['dist_good'] - bin_grouped['dist_bad']) * bin_grouped['woe']).sum()
            self.iv_dict[col] = iv

        return self

    def transform(self, X):
        X_woe = X.copy()
        for col, woe_map in self.woe_dict.items():
            if woe_map == 'constant':
                X_woe[col] = 0.0
                continue

            try:
                bins = pd.qcut(X_woe[col], q=self.bins, duplicates='drop')
            except ValueError:
                bins = pd.cut(X_woe[col], bins=min(self.bins, X_woe[col].nunique()), duplicates='drop')

            mapped = bins.map(woe_map)
            X_woe[col] = mapped.astype(float).fillna(0)  # Explicit astype to avoid categorical issues

        return X_woe

# Pipeline for proxy target (Task 4)
target_pipeline = Pipeline([
    ('time_extractor', TimeFeatureExtractor()),
    ('rfm', RFMTransformer()),
    ('cluster', ClusterTransformer())
])

# Pipeline for feature engineering (Task 3)
feature_pipeline = Pipeline([
    ('time_extractor', TimeFeatureExtractor()),
    ('aggregator', FeatureAggregator()),
    ('custom_scaler', CustomScaler())
])

if __name__ == "__main__":
    # Load raw data
    df = pd.read_csv("data/raw/data.csv")  # Replace with 'Train.csv' if that's the filename

    # Generate proxy target
    target_df = target_pipeline.fit_transform(df)
    target_df = target_df[['CustomerId', 'is_high_risk']]

    # Generate features
    features_df = feature_pipeline.fit_transform(df)

    # Merge features and target
    processed_df = pd.merge(features_df, target_df, on='CustomerId')

    # Apply WOE transformation and feature selection using IV
    X = processed_df.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = processed_df['is_high_risk']
    woe_transformer = CustomWOE(bins=5)
    woe_transformer.fit(X, y)
    X_woe = woe_transformer.transform(X)

    # Get Information Value (IV) and select features with IV > 0.02
    iv_df = pd.DataFrame({'Variable_Name': list(woe_transformer.iv_dict.keys()), 
                          'Information_Value': list(woe_transformer.iv_dict.values())})
    selected_features = iv_df[iv_df['Information_Value'] > 0.02]['Variable_Name'].tolist()
    X_selected = X_woe[selected_features]

    # Final dataframe
    final_df = pd.concat([processed_df['CustomerId'], X_selected, y], axis=1)

    # Save processed data
    final_df.to_csv("data/processed/processed_data.csv", index=False)

    # Optional: Print cluster means for verification
    rfm_temp = target_pipeline.steps[1][1].transform(target_pipeline.steps[0][1].transform(df))  # Get RFM
    rfm_with_cluster = target_pipeline.steps[2][1].transform(rfm_temp)  # Get clusters
    print("Cluster Means for Verification:\n", rfm_with_cluster.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean())

    print("Processed data saved to ../data/processed/processed_data.csv")
    print("Selected features based on IV:", selected_features)