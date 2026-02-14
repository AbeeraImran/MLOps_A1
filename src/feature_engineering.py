import pandas as pd
import os

input_path = 'data/processed/titanic_processed.csv'
output_dir = 'data/features'
output_file = os.path.join(output_dir, 'titanic_features.csv')

os.makedirs(output_dir, exist_ok=True)

print("Loading processed data...")
df = pd.read_csv(input_path)


print("Generating 'FamilySize' feature...")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

df.to_csv(output_file, index=False)
print(f"Feature engineering complete. Data saved to {output_file}")