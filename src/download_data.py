import os
import pandas as pd

raw_data_path = 'data/raw'
os.makedirs(raw_data_path, exist_ok=True)

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
print(f"Downloading data from {url}...")

try:
    df = pd.read_csv(url)
    output_file = os.path.join(raw_data_path, 'titanic.csv')
    df.to_csv(output_file, index=False)
    print(f"Data saved successfully to {output_file}")
except Exception as e:
    print(f"Error downloading data: {e}")