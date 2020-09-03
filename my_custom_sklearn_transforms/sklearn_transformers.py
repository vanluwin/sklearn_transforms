from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np

class DropColumns(BaseEstimator, TransformerMixin):
    """Elimina Colunas não necessárias para o modelo"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        data = x.copy()

        return data.drop(labels=self.columns, axis='columns')

class InsertColumns(BaseEstimator, TransformerMixin):
    """Elimina Colunas não necessárias para o modelo"""

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        data = x.copy()

        data['REPROVACOES_H'] = data['REPROVACOES_DE'] + data['REPROVACOES_EM']
        data['REPROVACOES_E'] = data['REPROVACOES_MF'] + data['REPROVACOES_GO']
        data['MEDIA_H'] = (data['NOTA_DE'] + data['NOTA_EM'])/2
        data['MEDIA_E'] = (data['NOTA_MF'] + data['NOTA_GO'])/2

        return data

class ReplaceMissingValues(BaseEstimator, TransformerMixin):
    """ Substitui uma coluna com a média e a outra com zeros"""

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        data = x.copy()

        ingles_count = data[[
            'INGLES', 'PERFIL'
        ]].groupby('PERFIL').agg(lambda x:x.value_counts().index[0])

        data['INGLES'] = data.apply(
            lambda row: ingles_count.loc[row['PERFIL']]['INGLES'] if np.isnan(row['INGLES']) else row['INGLES'],
            axis=1
        )

        nota_go_mean = data[['NOTA_GO', 'PERFIL']].groupby(['PERFIL']).mean()

        data['NOTA_GO'] = data.apply(
            lambda row: nota_go_mean.loc[row['PERFIL']]['NOTA_GO'] if np.isnan(row['NOTA_GO']) else row['NOTA_GO'],
            axis=1
        )

        return data
