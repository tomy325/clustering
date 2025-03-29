import pandas as pd

def calcular_porcentajes(path_csv):
    """
    Carga un archivo CSV con una columna 'filter' que contiene tres valores separados por guiones bajos
    y calcula los porcentajes de aparición de cada valor en las categorías: type, speed y duration.

    Parámetros:
    - path_csv: ruta al archivo CSV

    Retorna:
    - Un diccionario con los porcentajes de cada categoría
    """
    df = pd.read_csv(path_csv)

    # Separar la columna 'filter'
    df[['type', 'speed', 'duration']] = df['filter'].str.split('_', expand=True)

    # Calcular porcentajes
    type_pct = df['type'].value_counts(normalize=True) * 100
    speed_pct = df['speed'].value_counts(normalize=True) * 100
    duration_pct = df['duration'].value_counts(normalize=True) * 100

    return {
        'type': type_pct.round(2).to_dict(),
        'speed': speed_pct.round(2).to_dict(),
        'duration': duration_pct.round(2).to_dict()
    }

def mostrar_porcentajes(porcentajes):
    print("\n📊 Distribución de categorías:\n")
    
    for categoria, valores in porcentajes.items():
        print(f"🔹 {categoria.capitalize()}:")
        for clave, valor in valores.items():
            print(f"   - {clave}: {valor:.2f}%")
        print()  # Línea en blanco entre categorías

# Ejecutar
archivo = r"dosclusters24-03\mean_1.csv"  
porcentajes = calcular_porcentajes(archivo)
mostrar_porcentajes(porcentajes)
