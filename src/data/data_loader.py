import pandas as pd

def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def save_data(df, data_path):
    df.to_csv(data_path, index=False)


