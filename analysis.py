import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Load Data
print("Loading data...")
df = pd.read_csv('bank-full.csv', sep=';')

# 2. Data Preprocessing
print("Preprocessing data...")
# Convert target variable 'y' to binary
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Encoding categorical variables
le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 3. Feature Correlation Analysis (Requirement 2)
print("Analyzing feature correlations...")
plt.figure(figsize=(15, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.close()

# Specifically check correlations with 'y'
y_corr = correlation_matrix['y'].sort_values(ascending=False)
print("\nCorrelation with target variable 'y':")
print(y_corr)

# 4. Marketing Campaign Analysis (Requirement 3)
print("\nAnalyzing marketing campaign effectiveness...")
# Analyze call duration vs subscription
plt.figure(figsize=(10, 6))
sns.boxplot(x='y', y='duration', data=pd.read_csv('bank-full.csv', sep=';'))
plt.title('Call Duration vs Subscription')
plt.savefig('duration_vs_subscription.png')
plt.close()

# 5. Predictive Modeling (Requirement 1)
print("\nBuilding predictive model...")
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance from Model
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nTop 10 Important Features for Prediction:")
print(feature_importance_df.head(10))

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance for Subscribing Term Deposit')
plt.savefig('feature_importance.png')
plt.close()

print("\nAnalysis complete. Results and plots saved.")
