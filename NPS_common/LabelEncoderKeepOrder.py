# adapted from https://stackoverflow.com/questions/51308994/python-sklearn-determine-the-encoding-order-of-labelencoder

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import column_or_1d

class LabelEncoderKeepOrder(LabelEncoder):

    def fit(self, y):
        y = column_or_1d(y, warn=True)
        try:
            import pandas as pd
            self.classes_ = pd.Series(y).unique()
        except:
            print("pandas not found. skipping unique")
            self.classes = y
        return self
