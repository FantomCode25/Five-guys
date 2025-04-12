import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Generate dummy training data (replace with real data if available)
# This creates random data with 100 samples and 7 features (matching your input)
dummy_data = np.random.rand(100, 7)

# Create and fit scalers
stand_scaler = StandardScaler().fit(dummy_data)
minmax_scaler = MinMaxScaler().fit(dummy_data)

# Save scalers to files
with open('standscaler.pkl', 'wb') as f:
    pickle.dump(stand_scaler, f)

with open('minmaxscaler.pkl', 'wb') as f:
    pickle.dump(minmax_scaler, f)

print("Scalers created successfully!")