import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline

# ✅ Load CSV instead of df.pkl
df = pd.read_csv('laptop_data.csv')

# Features / Target
X = df.drop('Price', axis=1)
y = df['Price']

# Detect categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Transformer
transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Model (latest compatible)
model = DecisionTreeRegressor(random_state=42)

# Pipeline
pipe = Pipeline([
    ('transformer', transformer),
    ('model', model)
])

# Train model
pipe.fit(X, y)

# Save NEW modern pickle
pickle.dump(pipe, open('pipe.pkl', 'wb'))

print("✅ Model rebuilt successfully using latest libraries")
