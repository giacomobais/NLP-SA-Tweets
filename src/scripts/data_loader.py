import pandas as pd

def load_data(data_path):
    df = pd.read_csv(data_path, encoding='ISO-8859-1')
    return df



