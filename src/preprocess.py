import pandas as pd
import os

input_path = 'data/raw/titanic.csv'
output_dir = 'data/processed'
output_file = os.path.join(output_dir, 'titanic_processed.csv')

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_path)

# 3. Handle Missing Values
df['Age'] = df['Age'].fillna(df['Age'].median())
# Fill missing Embarked with the mode (most common value)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Fill missing Fare with median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# 4. Encode Categorical Variables
# Convert Sex to numbers: male=0, female=1
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Convert Embarked to numbers using One-Hot Encoding
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# 5. Drop columns we won't use for this simple model
# (Name, Ticket, and Cabin are hard to process simply)
df = df.drop(columns=['Name', 'Ticket', 'Cabin'])

# 6. Save Processed Data
df.to_csv(output_file, index=False)
print(f"Preprocessing complete. Data saved to {output_file}")