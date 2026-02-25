import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# Modelos
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

# Ignorar warnings molestos de convergencia para mantener limpia la salida
warnings.filterwarnings("ignore")


# =============================================================================
# 1. FUNCIÓN DE CARGA DE DATOS (INTEGRADA)
# =============================================================================
def LoadData(file_path):
    print(f"📂 Leyendo archivo: {file_path} ...")

    # 1. Cargar con liac-arff
    try:
        with open(file_path, 'r') as f:
            dataset = arff.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: No se encuentra el archivo en {file_path}")
        return None, None

    col_names = [attr[0] for attr in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=col_names)

    # 2. Limpieza básica
    df.replace([None], np.nan, inplace=True)
    filename = file_path.lower()

    # --- CORRECCIONES ESPECÍFICAS ---
    # CASO US CRIME: Eliminar ID inútil
    if 'crime' in filename and 'communityname' in df.columns:
        df = df.drop(columns=['communityname'])

    # CASO BOSTON: Arreglar columnas que cargan mal
    if 'boston' in filename:
        cols_to_fix = ['CHAS', 'RAD']
        for col in cols_to_fix:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. FORZADO NUMÉRICO (SOLO REGRESIÓN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. BORRAR COLUMNAS VACÍAS
    # Si una columna es 100% NaN (como 'Sex' en Abalone al forzar numérico), la borramos.
    df.dropna(axis=1, how='all', inplace=True)

    # 5. Imputación de nulos restantes
    if df.isnull().sum().sum() > 0:
        print(f"   ⚠️ Imputando {df.isnull().sum().sum()} valores nulos con la media...")
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(df)
        df = pd.DataFrame(data_imputed, columns=df.columns)

    # 6. Separar X e y
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"   ✅ Datos cargados. X: {X.shape}, y: {y.shape}")
    return X, y


# =============================================================================
# 2. CONFIGURACIÓN DEL ZOO DE MODELOS (PERFIL "LIGHTWEIGHT" TFG)
# =============================================================================
def get_model_pool(random_state=42):
    """
    Devuelve un diccionario con modelos configurados exactamente como en el EA.
    Objetivo: Velocidad, baja varianza y generalización (evitar overfitting).
    """
    return {
        # --- 1. Árboles y Ensembles (Configuración Rápida) ---

        # HGB: Tu configuración optimizada para convergencia rápida
        'HGB': HistGradientBoostingRegressor(
            max_iter=50,  # Bajado de 100
            max_depth=3,  # Árboles cortos
            early_stopping=True,
            scoring='loss',  # Cálculo rápido
            n_iter_no_change=5,  # Paciencia baja
            random_state=random_state
        ),

        # ExtraTrees: Más aleatoriedad, hojas grandes para suavizar ruido
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=50,  # Menos árboles
            min_samples_leaf=20,  # Hojas grandes = menos nodos = más rápido
            n_jobs=-1,  # Usa todos los núcleos en este análisis
            random_state=random_state
        ),

        # RandomForest: Misma lógica que ET para comparar peras con peras
        'RandomForest': RandomForestRegressor(
            n_estimators=50,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state
        ),

        # DT Simple: Un solo árbol débil (base de referencia)
        'DT-Simple': DecisionTreeRegressor(
            max_depth=5,
            random_state=random_state
        ),

        # --- 2. Lineales (Estabilidad / "El Ancla") ---

        # Ridge: El complemento perfecto para los árboles
        'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),

        # Lasso: Selección de características implícita
        'Lasso': make_pipeline(StandardScaler(), Lasso(alpha=0.1)),

        # Huber: Robusto a outliers (útil en drift abrupto con ruido)
        'Huber': make_pipeline(StandardScaler(), HuberRegressor(max_iter=100)),

        # --- 3. Instancias / Geométricos (Especialistas Locales) ---

        # KNN: Captura relaciones locales que los árboles globales pierden
        'KNN-5': make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),

        # MLP: Red neuronal muy pequeña (similar a una regresión logística vitaminada)
        'MLP': make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(30,), max_iter=200, random_state=random_state)
        )
    }

# =============================================================================
# 3. MOTOR DE ANÁLISIS
# =============================================================================
def ejecutar_analisis_completo(file_path):
    # A. Carga
    X, y = LoadData(file_path)
    if X is None: return

    # B. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # C. Entrenamiento Individual
    model_pool = get_model_pool()
    preds_dict = {}
    errors_dict = {}  # Residuos con signo (y - pred) para correlación
    sq_errors_dict = {}  # Errores cuadráticos para RMSE y Oráculo

    results_individual = []

    print("\n🚀 Entrenando Zoo de Modelos...")
    for name, model in model_pool.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            preds_dict[name] = pred

            # Métricas
            residuo = y_test - pred
            errors_dict[name] = residuo
            sq_errors_dict[name] = residuo ** 2

            rmse = np.sqrt(np.mean(residuo ** 2))
            results_individual.append({'Model': name, 'RMSE': rmse})
            print(f"   -> {name}: {rmse:.4f}")

        except Exception as e:
            print(f"   ❌ Fallo en {name}: {e}")

    # Mejor modelo individual (Baseline)
    df_ind = pd.DataFrame(results_individual).sort_values('RMSE')
    best_single_rmse = df_ind.iloc[0]['RMSE']
    best_single_name = df_ind.iloc[0]['Model']
    print(f"\n🏆 Mejor Individual: {best_single_name} (RMSE: {best_single_rmse:.4f})")

    # --- D. ANÁLISIS DE CORRELACIÓN (NUEVO) ---
    print("\n🔥 Generando Matriz de Correlación de Residuos...")
    matriz_corr = plot_correlation_matrix(preds_dict, y_test, title=file_path.split('/')[-1])

    # (Opcional) Imprimir parejas con menor correlación (Mayor diversidad)
    print("   Parejas más diversas (Menor correlación):")
    corrs = matriz_corr.unstack().sort_values()
    # Filtramos para quitar autocorrelaciones (1.0) y duplicados
    unique_corrs = corrs[(corrs < 0.99) & (corrs.index.get_level_values(0) < corrs.index.get_level_values(1))]
    print(unique_corrs.head(5).to_string())

    # ---------------------------------------------------------
    # E. VOTING REGRESSOR (Benchmark Real)
    # ---------------------------------------------------------
    print("\n⚖️ Evaluando Voting Regressor (Ensemble estático de todos)...")
    voting_clf = VotingRegressor([
        (name, model) for name, model in model_pool.items() if name in preds_dict
    ])
    voting_clf.fit(X_train, y_train)
    voting_pred = voting_clf.predict(X_test)
    voting_rmse = np.sqrt(mean_squared_error(y_test, voting_pred))
    print(f"   -> Voting Regressor RMSE: {voting_rmse:.4f}")

    # ---------------------------------------------------------
    # F. ORÁCULO COMBINATORIO (Mejor Trío)
    # ---------------------------------------------------------
    print("\n🔍 Buscando el Mejor Trío (Oráculo)...")
    trios = list(itertools.combinations(preds_dict.keys(), 3))
    trio_results = []

    for trio in trios:
        m1, m2, m3 = trio

        # Matriz de errores cuadráticos del trío (N_muestras x 3)
        trio_sq_errors = np.column_stack((
            sq_errors_dict[m1],
            sq_errors_dict[m2],
            sq_errors_dict[m3]
        ))

        # ORÁCULO: Para cada instancia, tomamos el error del MEJOR de los 3
        min_sq_errors = np.min(trio_sq_errors, axis=1)
        oracle_rmse = np.sqrt(np.mean(min_sq_errors))

        # Mejora porcentual sobre el mejor individual
        improvement = 100 * (1 - oracle_rmse / best_single_rmse)

        trio_results.append({
            'Trio': f"{m1} + {m2} + {m3}",
            'Oracle_RMSE': oracle_rmse,
            'Mejora_%': improvement,
            'Members': trio
        })

    df_trio = pd.DataFrame(trio_results).sort_values('Oracle_RMSE')
    best_trio = df_trio.iloc[0]

    print("\n🥇 TOP 3 MEJORES TRÍOS (Complementariedad):")
    print(df_trio.head(3).to_string(index=False))

    # ---------------------------------------------------------
    # G. VISUALIZACIÓN DEL MEJOR TRÍO
    # ---------------------------------------------------------
    m1, m2, m3 = best_trio['Members']
    err1 = errors_dict[m1]
    err2 = errors_dict[m2]
    err3 = errors_dict[m3]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # M1 vs M2
    axes[0].scatter(err1, err2, alpha=0.4, s=15, c='blue')
    axes[0].set_xlabel(f"Residuos {m1}")
    axes[0].set_ylabel(f"Residuos {m2}")
    axes[0].set_title(f"{m1} vs {m2}")
    axes[0].grid(True, alpha=0.3)

    # M1 vs M3
    axes[1].scatter(err1, err3, alpha=0.4, s=15, c='green')
    axes[1].set_xlabel(f"Residuos {m1}")
    axes[1].set_ylabel(f"Residuos {m3}")
    axes[1].set_title(f"{m1} vs {m3}")
    axes[1].grid(True, alpha=0.3)

    # M2 vs M3
    axes[2].scatter(err2, err3, alpha=0.4, s=15, c='orange')
    axes[2].set_xlabel(f"Residuos {m2}")
    axes[2].set_ylabel(f"Residuos {m3}")
    axes[2].set_title(f"{m2} vs {m3}")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f"Análisis del Ganador: {best_trio['Trio']} (Mejora Teórica: {best_trio['Mejora_%']:.2f}%)",
                 fontsize=16)
    plt.tight_layout()
    plt.show()  #

    print("\n✅ ANÁLISIS FINALIZADO.")


def plot_correlation_matrix(preds_dict, y_test, title="Matriz de Correlación de Residuos"):
    """
    Calcula y grafíca la correlación de Pearson entre los ERRORES de los modelos.
    IMPORTANTE: No correlacionamos las predicciones, sino los residuos (y - y_pred).
    Si dos modelos fallan en los mismos puntos, tienen alta correlación positiva (Malo).
    """
    import seaborn as sns

    # 1. Calcular Residuos (Errores con signo)
    # Si y_true es 100 y pred es 90, residuo = 10.
    # Si y_true es 100 y pred es 110, residuo = -10.
    residuals_dict = {name: y_test - pred for name, pred in preds_dict.items()}
    df_residuals = pd.DataFrame(residuals_dict)

    # 2. Calcular Matriz de Correlación
    corr_matrix = df_residuals.corr(method='pearson')

    # 3. Graficar Heatmap
    plt.figure(figsize=(10, 8))

    # Máscara para ocultar la mitad superior (opcional, queda más limpio)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,  # Mostrar números
        fmt=".2f",  # 2 decimales
        cmap='coolwarm_r',  # Rojo=Alto(Malo), Azul=Bajo(Bueno) -> Invertido '_r' para que rojo sea alerta
        vmin=-1, vmax=1,  # Escala fija
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    plt.title(f"Diversidad del Ensemble: {title}", fontsize=14)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return corr_matrix


# =============================================================================
# ZONA DE EJECUCIÓN (MODIFICA AQUÍ LA RUTA)
# =============================================================================
if __name__ == "__main__":
    DATASET_PATH = "../data/regression/elevators.arff"

    # Ejecutar
    ejecutar_analisis_completo(DATASET_PATH)