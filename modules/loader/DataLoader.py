import pandas as pd

class DataLoader:

    def load_from_csv(self, file_path = './data/sbrt_codebook_coded5.csv'):
        return pd.read_csv(file_path)