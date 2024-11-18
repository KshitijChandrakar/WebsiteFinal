import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer

file_path = 'titanic.csv'
df = pd.read_csv(file_path)
print("Dataset Information:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print("\nNumerical Columns:", numerical_cols)
print("Categorical Columns:", categorical_cols)
scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nFirst 5 rows after scaling numerical features:")
print(df_scaled.head())
df_encoded = pd.get_dummies(df_scaled, columns=categorical_cols)

print("\nFirst 5 rows after one-hot encoding:")
print(df_encoded.head())

print("\nComparison of the original and preprocessed dataset:")
print("Original Dataset (First 5 rows):\n", df.head())
print("Preprocessed Dataset (First 5 rows):\n", df_encoded.head())

print("\nMemory usage before preprocessing: {:.2f} KB".format(df.memory_usage(deep=True).sum() / 1024))
print("Memory usage after preprocessing: {:.2f} KB".format(df_encoded.memory_usage(deep=True).sum() / 1024))