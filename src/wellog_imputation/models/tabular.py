"""Tabular imputers: MEAN, Random Forest, XGBoost, MICE.

All of them treat a window as an unordered ``(N*T, C)`` matrix: a missing entry is
predicted from the OTHER logs co-located at the same depth.  No depth ordering and no
neighbouring well is used.

Common interface (see :mod:`wellog_imputation.evaluation.harness`)::

    m.fit(Xtr_in, Xtr_true, Mtr)
    m.impute(Xev_in) -> (N, T, C)
"""
import numpy as np

__all__ = ["MeanImputer", "TabularImputer", "MICEImputer"]


class MeanImputer:
    """Zero-skill floor: predict the training mean of each channel.

    Inputs are standardised per fold, so the training mean is 0 in the working space.
    """

    name = "MEAN"

    def fit(self, Xin, Xtrue, M):
        pass

    def impute(self, Xev):
        Y = Xev.copy()
        Y[np.isnan(Y)] = 0.0           # 0 = the training mean in normalised space
        return Y


class TabularImputer:
    """One regressor per log: predict the target log from the others at the same depth.

    Parameters
    ----------
    kind : {'rf', 'xgb'}
        Random forest or gradient boosting.  Hyperparameters are listed in
        configs/models/tabular.yaml and set in :meth:`_new`.
    """

    def __init__(self, kind="rf", train_on="intact"):
        # train_on="intact"   : original protocol, fit on complete training rows;
        # train_on="corrupted": revision R1 (Reviewer 2, comment 3) -- predictors taken
        #   from the curriculum-corrupted input Xin (NaN where masked), targets from the
        #   intact windows, so the model sees missing predictors during training.  Only
        #   meaningful for XGBoost, which handles NaN natively.
        self.kind = kind; self.name = kind.upper() + ("-MISS" if train_on == "corrupted" else "")
        self.train_on = train_on
        self.reg = {}

    def _new(self):
        if self.kind == "rf":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=200, n_jobs=-1, max_depth=None,
                                         min_samples_leaf=2, random_state=0)
        if self.kind == "xgb":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                                subsample=0.8, n_jobs=-1, random_state=0)
        # Revision R1 (Reviewer 1, comment 8): conventional baselines.
        if self.kind == "ridge":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        if self.kind == "knn":
            from sklearn.neighbors import KNeighborsRegressor
            return KNeighborsRegressor(n_neighbors=10, weights="distance", n_jobs=-1)
        raise ValueError(self.kind)

    def fit(self, Xin, Xtrue, M):
        flat = Xtrue.reshape(-1, Xtrue.shape[-1])          # (N*T, C), intact windows
        C = flat.shape[1]
        flat_in = Xin.reshape(-1, C) if self.train_on == "corrupted" else flat
        for c in range(C):
            others = [j for j in range(C) if j != c]
            Xf = flat_in[:, others]; y = flat[:, c]
            if self.train_on == "corrupted":
                ok = np.isfinite(y) & np.any(np.isfinite(Xf), 1)   # NaN predictors kept
            else:
                ok = np.all(np.isfinite(Xf), 1) & np.isfinite(y)
            reg = self._new(); reg.fit(Xf[ok], y[ok])
            self.reg[c] = reg

    def impute(self, Xev):
        Y = Xev.copy()
        N, T, C = Y.shape
        flat = Y.reshape(-1, C)
        for c in range(C):
            others = [j for j in range(C) if j != c]
            miss = np.isnan(flat[:, c])
            if not miss.any():
                continue
            feats = flat[miss][:, others]
            if self.kind in ("rf", "ridge", "knn"):   # cannot consume NaN -> 0 (the mean)
                feats = np.nan_to_num(feats, nan=0.0)
            pred = self.reg[c].predict(feats)
            flat[miss, c] = pred
        return flat.reshape(N, T, C)


class MICEImputer:
    """Canonical MICE (chained equations): sklearn ``IterativeImputer`` + ``BayesianRidge``.

    Data handling is identical to :class:`TabularImputer` (fit on the intact training
    windows of the fold); the difference is a single chained imputer with a shared
    BayesianRidge instead of one independent regressor per log.

    Determinism: ``sample_posterior=False`` (point imputation, no sampling) with a fixed
    ``random_state``.

    Blackout fallback: for a row whose predictors are themselves missing,
    ``IterativeImputer`` falls back to ``initial_strategy='mean'`` -- means ESTIMATED ON
    THE TRAINING SET at fit time.  Being a regularised linear model, MICE therefore
    degrades gracefully towards the mean instead of extrapolating off-support (in
    contrast with XGBoost).  This is expected behaviour and is left uncorrected.
    """

    name = "MICE"

    def __init__(self, max_iter=20, tol=1e-3, random_state=0):
        self.max_iter = max_iter; self.tol = tol; self.random_state = random_state
        self.imp = None

    def fit(self, Xin, Xtrue, M):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        flat = Xtrue.reshape(-1, Xtrue.shape[-1])          # (N*T, C), as for RF/XGB
        ok = np.all(np.isfinite(flat), 1)
        self.imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=False,
                                    max_iter=self.max_iter, tol=self.tol,
                                    initial_strategy="mean", random_state=self.random_state)
        self.imp.fit(flat[ok])                             # training fold only (anti-leakage)

    def impute(self, Xev):
        Y = Xev.copy()
        N, T, C = Y.shape
        flat = Y.reshape(-1, C)
        out = self.imp.transform(flat)                     # NaN filled by chained equations
        return out.reshape(N, T, C).astype(Y.dtype)


class XGBChainImputer:
    """XGBoost inside sklearn's ``IterativeImputer`` (chained equations), revision R1.

    Tests the sensitivity of the XGBoost results to the imputation order (Reviewer 2,
    comment 3): ``order`` is passed to ``imputation_order`` ('ascending' = logs with the
    fewest missing values first, 'descending' = the reverse).  Fitted on the intact
    training rows, like MICE; at prediction the chained equations start from the
    training means and iterate.
    """

    def __init__(self, order="ascending", max_iter=10, random_state=0):
        self.order = order; self.max_iter = max_iter; self.random_state = random_state
        self.name = f"XGB-CHAIN-{order[:3].upper()}"
        self.imp = None

    def fit(self, Xin, Xtrue, M):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from xgboost import XGBRegressor
        flat = Xtrue.reshape(-1, Xtrue.shape[-1])
        ok = np.all(np.isfinite(flat), 1)
        est = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                           subsample=0.8, n_jobs=-1, random_state=0)
        self.imp = IterativeImputer(estimator=est, max_iter=self.max_iter,
                                    initial_strategy="mean", imputation_order=self.order,
                                    random_state=self.random_state)
        self.imp.fit(flat[ok])

    def impute(self, Xev):
        Y = Xev.copy(); N, T, C = Y.shape
        return self.imp.transform(Y.reshape(-1, C)).reshape(N, T, C).astype(Y.dtype)
