import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load and preprocess training data
df_train = pd.read_csv("C:/Users/neera/Downloads/train_LZdllcl.csv")

# Initialize label encoder
encoder = LabelEncoder()

# Fit and transform columns
df_train['department'] = encoder.fit_transform(df_train['department'])
df_train['region'] = encoder.fit_transform(df_train['region'])
df_train['education'] = encoder.fit_transform(df_train['education'])
df_train['gender'] = encoder.fit_transform(df_train['gender'])
df_train['recruitment_channel'] = encoder.fit_transform(df_train['recruitment_channel'])

print(df_train.head())

# Define features and target
X = df_train.drop(columns='is_promoted')
y = df_train['is_promoted']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.head())

# Initialize and train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Predictions and evaluation
pred = model.predict(X_test)
print(pred)

cm = confusion_matrix(y_test, pred)
print(cm)
print(accuracy_score(y_test, pred))

# Load and preprocess test data
df_test = pd.read_csv("C:/Users/neera/Downloads/test_2umaH9m.csv")

# Use the same encoder fitted on training data to avoid issues with unseen labels
df_test['department'] = encoder.transform(df_test['department'])
df_test['region'] = encoder.transform(df_test['region'])
df_test['education'] = encoder.transform(df_test['education'])
df_test['gender'] = encoder.transform(df_test['gender'])
df_test['recruitment_channel'] = encoder.transform(df_test['recruitment_channel'])

print(df_test.head())

# Prepare features for prediction
X_test_final = df_test.drop(columns=['employee_id'])  # Drop non-feature columns

print("Test Data: ", X_test_final)

# Make predictions on the test set
pred_final = model.predict(X_test_final)

# Prepare submission DataFrame
submission = pd.DataFrame({
    'employee_id': df_test['employee_id'],
    'is_promoted': pred_final
})

# Save to CSV
submission.to_csv('predictions3.csv', index=False)
