import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Setup Paths
input_path = 'data/features/titanic_features.csv'
model_path = 'models/model.pkl'

os.makedirs('models', exist_ok=True)

# 2. Load Data
print("Loading data...")
df = pd.read_csv(input_path)

# 3. Separate Features (X) and Target (y)
y = df['Survived']
# Drop target and irrelevant columns
X = df.drop(columns=['Survived', 'PassengerId'])

# 4. Split Data (80% train, 20% test)
# We set random_state=42 to ensure the split is the same every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
print("Training Random Forest model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Save the Model
with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

print(f"Model saved to {model_path}")