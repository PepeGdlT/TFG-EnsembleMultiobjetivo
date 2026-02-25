"""Super-script: selección por complementariedad (Oráculo) + ajuste focalizado + guardado.

Objetivo por dataset (regresión):
  1) Entrenamiento rápido de un zoo lightweight.
  2) Predicción en validación y cálculo de residuos.
  3) Búsqueda del mejor trío (Oracle RMSE mínimo) y su mejora vs mejor modelo individual.
  4) RandomizedSearchCV SOLO sobre ese trío.
  5) Guardado del trío ajustado en /modelos_ajustados/<dataset>_best_models.pkl

Diseñado para integrarse con el resto del repo (usa src.utils.LoadData).
"""

from __future__ import annotations

import os
import glob
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Protocol, runtime_checkable

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.metrics import mean_squared_error

# Zoo models
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

from src.utils import LoadData

warnings.filterwarnings("ignore")


@runtime_checkable
class RegressorLike(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressorLike": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


DIRECTORIO_SALIDA = os.path.join(os.path.dirname(__file__), "..", "modelos_ajustados")
os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)


# -----------------------------
# Configuración del Zoo (lightweight)
# -----------------------------

def get_lightweight_model_zoo(random_state: int = 42) -> Dict[str, RegressorLike]:
    """Pool de modelos rápido y razonablemente diverso.

    Nota: intencionalmente NO hiperajustado aquí; esto es fase de preselección.
    """

    return {
        # Ensembles / Árboles
        "HGB": HistGradientBoostingRegressor(
            max_iter=60,
            max_depth=3,
            learning_rate=0.1,
            early_stopping=True,
            random_state=random_state,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=80,
            min_samples_leaf=10,
            bootstrap=False,
            n_jobs=-1,
            random_state=random_state,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=10,
            n_jobs=-1,
            random_state=random_state,
        ),
        "DT": DecisionTreeRegressor(max_depth=6, random_state=random_state),

        # Lineales
        "Ridge": make_pipeline(RobustScaler(), Ridge(alpha=1.0, solver="auto")),
        "Lasso": make_pipeline(RobustScaler(), Lasso(alpha=0.01, max_iter=5000, random_state=random_state)),
        "Huber": make_pipeline(RobustScaler(), HuberRegressor(max_iter=500)),

        # Vecinos
        "KNN": make_pipeline(
            RobustScaler(),
            KNeighborsRegressor(n_neighbors=11, weights="distance", p=2, n_jobs=1),
        ),

        # Red
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(40,),
                activation="relu",
                alpha=1e-4,
                max_iter=300,
                random_state=random_state,
                early_stopping=True,
            ),
        ),
    }


# -----------------------------
# Oráculo para tríos
# -----------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass(frozen=True)
class OracleTrioResult:
    members: Tuple[str, str, str]
    oracle_rmse: float
    best_single_name: str
    best_single_rmse: float
    improvement_pct: float


def find_best_oracle_trio(
    y_val: np.ndarray,
    preds: Dict[str, np.ndarray],
) -> OracleTrioResult:
    """Busca el trío con menor Oracle RMSE.

    Oracle RMSE = RMSE del error mínimo por instancia entre los modelos del trío.
    """

    # Baseline: mejor individual
    singles = {name: _rmse(y_val, p) for name, p in preds.items()}
    best_single_name = min(singles, key=singles.get)
    best_single_rmse = singles[best_single_name]

    # Precomputo errores cuadrados por modelo (más rápido)
    sq_err = {name: (y_val - p) ** 2 for name, p in preds.items()}

    best_members: Tuple[str, str, str] | None = None
    best_oracle = np.inf

    names = list(preds.keys())
    for a, b, c in itertools.combinations(names, 3):
        mat = np.column_stack([sq_err[a], sq_err[b], sq_err[c]])
        oracle_rmse = float(np.sqrt(np.mean(np.min(mat, axis=1))))
        if oracle_rmse < best_oracle:
            best_oracle = oracle_rmse
            best_members = (a, b, c)

    assert best_members is not None, "No hay suficientes modelos para formar tríos."
    improvement_pct = 100.0 * (1.0 - (best_oracle / best_single_rmse))

    return OracleTrioResult(
        members=best_members,
        oracle_rmse=float(best_oracle),
        best_single_name=best_single_name,
        best_single_rmse=float(best_single_rmse),
        improvement_pct=float(improvement_pct),
    )


# -----------------------------
# RandomizedSearchCV focalizado (solo para el trío)
# -----------------------------


def get_search_space(model_name: str, X_shape: Tuple[int, int], random_state: int = 42):
    """Espacios de búsqueda compactos para no destruir la 'esencia' del modelo."""
    n_samples, n_features = X_shape

    if model_name == "ExtraTrees":
        rango_features = [0.3, 0.5, 0.7, 0.9] if n_features <= 20 else [0.1, 0.2, 0.3, 0.5]
        return {
            "n_estimators": [50, 80, 120, 200],
            "max_depth": [None, 6, 10, 14],
            "min_samples_leaf": [3, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": rango_features,
        }

    if model_name == "RandomForest":
        rango_features = [0.3, 0.5, 0.7, 0.9] if n_features <= 20 else [0.1, 0.2, 0.3, 0.5]
        return {
            "n_estimators": [100, 150, 250],
            "max_depth": [None, 6, 10, 14],
            "min_samples_leaf": [3, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": rango_features,
        }

    if model_name == "HGB":
        return {
            "max_iter": [60, 100, 160],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_samples_leaf": [10, 20, 30],
            "l2_regularization": [0.0, 0.1, 1.0],
        }

    if model_name == "DT":
        return {
            "max_depth": [3, 5, 7, 10, None],
            "min_samples_leaf": [1, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10, 20],
        }

    if model_name == "Ridge":
        return {"ridge__alpha": np.logspace(-3, 3, 50).tolist()}

    if model_name == "Lasso":
        return {"lasso__alpha": np.logspace(-4, 1, 60).tolist()}

    if model_name == "Huber":
        return {
            "huberregressor__epsilon": [1.2, 1.35, 1.5, 1.75],
            "huberregressor__alpha": [0.0, 1e-5, 1e-4, 1e-3],
        }

    if model_name == "KNN":
        k_max = min(61, max(11, int(np.sqrt(n_samples))))
        return {
            "kneighborsregressor__n_neighbors": list(range(5, k_max + 1, 2)),
            "kneighborsregressor__weights": ["uniform", "distance"],
            "kneighborsregressor__p": [1, 2],
        }

    if model_name == "MLP":
        return {
            "mlpregressor__hidden_layer_sizes": [(20,), (40,), (60,), (40, 20)],
            "mlpregressor__alpha": [1e-5, 1e-4, 1e-3],
            "mlpregressor__learning_rate_init": [1e-4, 3e-4, 1e-3],
        }

    return None


def tune_model(
    model_name: str,
    model: RegressorLike,
    X: np.ndarray,
    y: np.ndarray,
    cv: KFold,
    random_state: int = 42,
) -> RegressorLike:
    space = get_search_space(model_name, X.shape, random_state=random_state)
    if not space:
        # fallback
        model.fit(X, y)
        return model

    # Presupuesto de búsqueda: pequeño por diseño
    n_iter = 25
    if model_name in {"ExtraTrees", "RandomForest"}:
        n_iter = 30

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        refit=True,
        verbose=0,
    )
    search.fit(X, y)
    return search.best_estimator_


# -----------------------------
# Pipeline por dataset
# -----------------------------


def process_dataset(
    dataset_path: str,
    output_dir: str = DIRECTORIO_SALIDA,
    random_state: int = 42,
    test_size_val: float = 0.30,
):
    dataset_name = os.path.basename(dataset_path).replace(".arff", "")

    X, y = LoadData(dataset_path)

    # Split de validación (solo para Oráculo)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size_val, random_state=random_state
    )

    zoo = get_lightweight_model_zoo(random_state=random_state)

    # Fase 1 + 2: entrenar rápido y generar residuos/preds
    preds: Dict[str, np.ndarray] = {}
    fitted_base: Dict[str, RegressorLike] = {}

    for name, model in zoo.items():
        try:
            m: RegressorLike = clone(model)  # type: ignore[assignment]
            m.fit(X_train, y_train)
            preds[name] = m.predict(X_val)
            fitted_base[name] = m
        except Exception:
            # si un modelo falla en algún dataset, lo saltamos (ej: convergencia)
            continue

    if len(preds) < 3:
        raise RuntimeError(f"No hay suficientes modelos válidos para {dataset_name} (solo {len(preds)}).")

    # Fase 3: Oráculo
    trio = find_best_oracle_trio(y_val=y_val, preds=preds)

    # Fase 4: ajuste focalizado (sobre todo el dataset; CV interna)
    cv = KFold(n_splits=10, shuffle=True, random_state=random_state)

    tuned_models: List[RegressorLike] = []
    for member in trio.members:
        base_est = zoo[member]
        tuned = tune_model(member, clone(base_est), X, y, cv=cv, random_state=random_state)  # type: ignore[arg-type]
        tuned_models.append(tuned)

    # Fase 5: guardado
    out_path = os.path.join(output_dir, f"{dataset_name}_best_models.pkl")
    joblib.dump(tuned_models, out_path)

    return {
        "dataset": dataset_name,
        "best_single": trio.best_single_name,
        "best_single_rmse": trio.best_single_rmse,
        "oracle_trio": " + ".join(trio.members),
        "oracle_rmse": trio.oracle_rmse,
        "oracle_improvement_pct": trio.improvement_pct,
        "saved_to": out_path,
    }


def main():
    datasets = glob.glob(os.path.join(os.path.dirname(__file__), "..", "data", "regression", "*.arff"))
    if not datasets:
        print("No se han encontrado datasets .arff en data/regression")
        return

    for ds in datasets:
        info = process_dataset(ds)
        print(
            f"[{info['dataset']}] Mejor individual: {info['best_single']} (RMSE={info['best_single_rmse']:.4f}) | "
            f"Oráculo trío: {info['oracle_trio']} (RMSE={info['oracle_rmse']:.4f}, mejora={info['oracle_improvement_pct']:.2f}%) | "
            f"Guardado: {info['saved_to']}"
        )


if __name__ == "__main__":
    main()
