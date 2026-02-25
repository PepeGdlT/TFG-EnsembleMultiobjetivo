import pandas as pd
import numpy as np
import os

# Intentamos importar la librería robusta
try:
    import arff
except ImportError:
    print("❌ ERROR CRÍTICO: Necesitas instalar 'liac-arff'.")
    print("   Ejecuta en terminal: pip install liac-arff")
    exit()

# Ajuste de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- CONFIGURACIÓN DE DATASETS POR CARPETA ---
# Usamos un diccionario para saber en qué subcarpeta buscar cada archivo
DATASETS = {
    "regression": [
        "abalone.arff",
        "boston.arff",
        "concrete.arff",
        "elevators.arff",
        "us_crime.arff",
        "cpu_act.arff",
        "ailerons.arff",
        "superconduct.arff"
    ],
    "classification": [
        "breastw.arff",
        "wine.arff",
        "sonar.arff",
        "ionosphere.arff",
        "glass.arff"
    ]
}


def cargar_arff_robusto(file_path):
    """Carga ARFF usando liac-arff y convierte a DataFrame limpio"""
    with open(file_path, 'r') as f:
        dataset = arff.load(f)

    col_names = [attr[0] for attr in dataset['attributes']]
    df = pd.DataFrame(dataset['data'], columns=col_names)
    df.replace([None], np.nan, inplace=True)
    return df


def inspeccionar_dataset(filename, subcarpeta):
    # Construimos la ruta incluyendo la subcarpeta (regression/classification)
    file_path = os.path.join(DATA_DIR, subcarpeta, filename)

    print("\n" + "=" * 60)
    print(f"📂 ANALIZANDO ({subcarpeta.upper()}): {filename}")
    print("=" * 60)

    try:
        if not os.path.exists(file_path):
            print(f"⚠️  ARCHIVO NO ENCONTRADO: {file_path}")
            return

        df = cargar_arff_robusto(file_path)
        n_rows, n_cols = df.shape
        print(f"📊 Dimensiones: {n_rows} instancias x {n_cols} atributos")

        # --- ANÁLISIS DEL TARGET ---
        target_name = df.columns[-1]
        target_col = df.iloc[:, -1]

        # Intentar convertir a número para el análisis
        try:
            target_numeric = pd.to_numeric(target_col)
            es_numerico_puro = True
        except:
            target_numeric = target_col
            es_numerico_puro = False

        n_unique = target_col.nunique()
        counts = target_col.value_counts(normalize=True)

        print(f"🎯 Target: '{target_name}'")
        print(f"   - Tipo detectado: {target_col.dtype}")
        print(f"   - Valores únicos: {n_unique}")

        # --- VEREDICTO MEJORADO ---
        es_clasificacion = False

        if not es_numerico_puro:
            print("   ✅ VEREDICTO: CLASIFICACIÓN (Etiquetas de texto).")
            es_clasificacion = True
        elif n_unique < 20:
            print("   ✅ VEREDICTO: CLASIFICACIÓN (Etiquetas numéricas / pocas clases).")
            es_clasificacion = True
        else:
            print("   ✅ VEREDICTO: REGRESIÓN (Continuo).")

        # --- SI ES CLASIFICACIÓN: MOSTRAR BALANCE ---
        if es_clasificacion:
            print("   ⚖️  Distribución de Clases (Balance):")
            for cls, prop in counts.items():
                print(f"      - Clase '{cls}': {prop:.2%}")

        # --- DETECCIÓN DE PROBLEMAS ---
        null_counts = df.isnull().sum().sum()
        features = df.iloc[:, :-1]
        cat_cols = features.select_dtypes(include=['object', 'category']).columns.tolist()

        print("\n🔍 Inspección de Features:")
        if null_counts > 0:
            print(f"   ⚠️ ALERTA: Hay {null_counts} valores NULOS.")
        else:
            print("   ✅ Limpio: No hay valores nulos.")

        if len(cat_cols) > 0:
            print(f"   ⚠️ ALERTA: Hay {len(cat_cols)} columnas CATEGÓRICAS (Texto).")
            print(f"      Ejemplo: {cat_cols[:3]}...")
        else:
            print("   ✅ Limpio: Todo numérico.")

    except Exception as e:
        print(f"❌ ERROR leyendo el archivo: {e}")


if __name__ == "__main__":
    print(f"Directorio Base de Datos: {DATA_DIR}")

    # Iteramos por tipo (regression/classification) y luego por archivo
    for tipo, lista_archivos in DATASETS.items():
        print(f"\n>>> PROCESANDO CARPETA: {tipo.upper()} <<<")
        for f in lista_archivos:
            inspeccionar_dataset(f, subcarpeta=tipo)