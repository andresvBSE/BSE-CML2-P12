import datetime
import pandas as pd
from typing import List

class DatesPreprocessor(object):
    def __init__(self):
        pass

    def preprocess(self, df_in: pd.DataFrame, feature_names: List[str]):
        self.feature_names = feature_names
        self.df_in = df_in
        return self._transform()

    def _transform(self):
        df_transformed = self.df_in.copy()
        for col in self.feature_names:
            df_transformed[col+'_year'] = df_transformed[col].dt.year
            df_transformed[col+'_month'] = df_transformed[col].dt.month
            df_transformed[col+'_day'] = df_transformed[col].dt.day
            df_transformed[col+'_weekday'] = df_transformed[col].dt.weekday
            df_transformed[col+'_hour'] = df_transformed[col].dt.hour
            df_transformed[col] = df_transformed[col].astype(int)
        return df_transformed
        
        
        
def reweight_proba(pi,q1=0.5,r1=0.5):
    r0 = 1-r1
    q0 = 1-q1
    tot = pi*(q1/r1)+(1-pi)*(q0/r0)
    w = pi*(q1/r1)
    w /= tot
    return w