from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from copulas.multivariate import GaussianMultivariate


@dataclass
class CopulaModel:
    columns: list[str]
    model: GaussianMultivariate | None = None
    random_state: int = 42

    def fit(self, df: pd.DataFrame) -> "CopulaModel":
        numeric = df[self.columns].dropna()
        if numeric.empty:
            raise ValueError("No complete rows available for copula training")
        np.random.seed(self.random_state)
        self.model = GaussianMultivariate()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(numeric)
        return self

    def sample(self, n: int) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not fitted - call fit() first")
        np.random.seed(self.random_state)
        synth = self.model.sample(n)
        return synth[self.columns].reset_index(drop=True)


def fit_copula(df: pd.DataFrame, columns: list[str], random_state: int = 42) -> CopulaModel:
    return CopulaModel(columns=columns, random_state=random_state).fit(df)