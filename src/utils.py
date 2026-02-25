import pandas as pd
import numpy as np
import arff
from sklearn.impute import SimpleImputer

def LoadData(file_path):
    # 1. Cargar con liac-arff
    with open(file_path, 'r') as f:
        dataset = arff.load(f)

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
    # Convertimos todo a números.
    # En Abalone, 'Sex' se convertirá enteramente en NaNs.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. BORRAR COLUMNAS VACÍAS (EL FIX PARA EL ERROR)
    # Si una columna es 100% NaN (como 'Sex' en Abalone), la borramos YA.
    # Así el imputer no se queja y los tamaños coinciden.
    df.dropna(axis=1, how='all', inplace=True)

    # 5. Imputación de nulos restantes
    # Si quedan huecos sueltos, los rellenamos con la media
    if df.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='mean')
        # Usamos un DataFrame nuevo para evitar conflictos de índices
        data_imputed = imputer.fit_transform(df)
        df = pd.DataFrame(data_imputed, columns=df.columns)

    # 6. Separar X e y
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y