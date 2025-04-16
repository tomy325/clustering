import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Cargar tu tabla resumen ===
df = pd.read_csv(r"resultados_pipeline_20250415_183251\resumen_mean_isi_spike.csv", index_col=0)

# === Filtros que representan combinaciones completas ===
# Se asume que los nombres de filtros completos tienen dos guiones bajos ("_")
filtros_completos = df.index[df.index.str.count('_') == 2]
df_filtros = df.loc[filtros_completos]

# === Categorías generales ===
categorias = ['ON', 'OF', 'fast', 'slow', 'sustained', 'transient']
df_categorias = df.loc[df.index.isin(categorias)]

# === Gráfico 1: Heatmap de filtros completos ===
plt.figure(figsize=(10, 8))
sns.heatmap(df_filtros, annot=True, cmap="Blues", fmt="d")
plt.title("Heatmap de combinaciones completas por cluster")
plt.ylabel("Filtro")
plt.xlabel("Cluster")
plt.tight_layout()
plt.show()

# === Gráfico 2: Barras agrupadas de categorías generales ===
df_categorias.T.plot(kind='bar', stacked=False, figsize=(10, 6))
plt.title("Conteo por categoría general y cluster")
plt.ylabel("Cantidad de elementos")
plt.xlabel("Cluster")
plt.tight_layout()
plt.show()

# === Gráfico 3: Barras agrupadas de filtros completos ===
df_filtros.T.plot(kind='bar', stacked=False, figsize=(12, 6))
plt.title("Conteo por filtro completo y cluster")
plt.ylabel("Cantidad de elementos")
plt.xlabel("Cluster")
plt.tight_layout()
plt.show()
