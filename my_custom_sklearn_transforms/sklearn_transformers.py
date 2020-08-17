from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class DropColumns(BaseEstimator, TransformerMixin):
    """Elimina Colunas não necessárias para o modelo"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self
    
    def transform(self, x):
        data = x.copy()

        return data.drop(labels=self.columns, axis='columns')

class MergeClasses(BaseEstimator, TransformerMixin):
    """Junta as classes alvo de classificação"""
    def __init__(self, target, new_value):
        self.target = target
        self.new_value = new_value

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        data = x.copy()

        return data.replace(self.target, self.new_value)

class ReplaceMissingValues(BaseEstimator, TransformerMixin):
    """ Substitui uma coluna com a média e a outra com zeros"""
    def __init__(self, mean_column, zero_column):
        self.mean_column = mean_column
        self.zero_column = zero_column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        data = x.copy()

        mean_imputer = SimpleImputer(strategy='mean')
        zero_imputer = SimpleImputer(strategy='constant', fill_value=0)

        data[[self.mean_column]] = mean_imputer.fit_transform(data[[self.mean_column]])
        data[[self.zero_column]] = zero_imputer.fit_transform(data[[self.zero_column]])

        return data
