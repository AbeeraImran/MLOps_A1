import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

input_path = 'data/features/titanic_features.csv'
model_path = 'models/model.pkl'
output_dir = 'results'
output_file = os.path.join(output_dir, 'predictions.csv')

os.makedirs(output_dir, exist_ok=True)


print("Loading data...")
df = pd.read_csv(input_path)
X = df.drop(columns=['Survived', 'PassengerId'])
y = df['Survived']

_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Loading trained model...")
with open(model_path, 'rb') as f:
    clf = pickle.load(f)

print("Generating predictions...")
y_pred = clf.predict(X_test)


results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
results_df.to_csv(output_file, index=False)

print(f"Predictions saved to {output_file}")