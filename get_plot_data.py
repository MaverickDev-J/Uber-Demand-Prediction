import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load processing data
print("Loading processing_data.csv...")
processing_data = pd.read_csv('processing_data.csv')
print(f"Processing data shape: {processing_data.shape}")

# Load trained scaler and KMeans model
print("Loading scaler and KMeans model...")
scaler = joblib.load('models/scaler.joblib')
kmeans = joblib.load('models/mb_kmeans.joblib')

# Extract coordinates and scale them
print("Scaling coordinates...")
coords = processing_data[['pickup_longitude', 'pickup_latitude']]
coords_scaled = scaler.transform(coords)

# Predict region using KMeans
print("Predicting regions...")
processing_data['region'] = kmeans.predict(coords_scaled)

# Check region distribution
region_counts = processing_data['region'].value_counts().sort_index()
print("\nRows per region (before sampling):")
print(region_counts)

# Sample 500 rows from each region
print("\nSampling 500 rows from each region...")
sampled_data = []

for region in range(30):  # Assuming 30 clusters (regions)
    region_data = processing_data[processing_data['region'] == region]
    
    if len(region_data) >= 500:
        sampled_region = region_data.sample(n=500, random_state=42)
        sampled_data.append(sampled_region)
        print(f"Region {region}: Sampled 500 rows from {len(region_data)} available")
    else:
        print(f"WARNING: Region {region} has only {len(region_data)} rows (need 500)")

# Combine sampled data
if len(sampled_data) == 30:
    plot_data = pd.concat(sampled_data, ignore_index=True)
    
    # Keep only required columns
    plot_data = plot_data[['pickup_longitude', 'pickup_latitude', 'region']]
    
    # Shuffle
    plot_data = plot_data.sample(frac=1, random_state=42).reset_index(drop=True)

    plot_data.to_csv('plot_data.csv', index=False)
    print("\n✅ Final plot_data.csv saved.")
    print(f"Final shape: {plot_data.shape}")
    print("Rows per region in final plot data:")
    print(plot_data['region'].value_counts().sort_index())
    
    # Preview
    print("\nSample of final data:")
    print(plot_data.head())

else:
    print(f"\n❌ ERROR: Could only sample from {len(sampled_data)} regions out of 30.")
    print("Some regions do not have at least 500 rows.")
