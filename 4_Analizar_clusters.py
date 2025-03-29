import pandas as  pd
import matplotlib.pyplot as plt
import os

# Cargar el DataFrame actualizado
df = pd.read_csv('dosclusters''.csv')
df_copy=df

# Obtener los valores únicos de clusters
unique_clusters1 = df['clusters_mean_isi_spike'].unique()
unique_clusters2 = df['clusters_isi'].unique()
unique_clusters3 = df['clusters_spike'].unique()



# Definir una carpeta para guardar los archivos
output_folder = input("Ingrese el nombre de la carpeta: ")
os.makedirs(output_folder, exist_ok=True)  # Crea la carpeta si no existe

# Guardar archivos en la carpeta correspondiente
for cluster in unique_clusters1:  
    filtered_df = df_copy[df_copy['clusters_mean_isi_spike'] == cluster][['filter', 'clusters_mean_isi_spike','clusters_isi','clusters_spike']]
    filtered_df.to_csv(os.path.join(output_folder, f'mean_{cluster}.csv'), index=False)

for cluster in unique_clusters2:  
    filtered_df = df_copy[df_copy['clusters_isi'] == cluster][['filter', 'clusters_mean_isi_spike','clusters_isi','clusters_spike']]
    filtered_df.to_csv(os.path.join(output_folder, f'isi_{cluster}.csv'), index=False)

for cluster in unique_clusters3:  
    filtered_df = df_copy[df_copy['clusters_spike'] == cluster][['filter', 'clusters_mean_isi_spike','clusters_isi','clusters_spike']]
    filtered_df.to_csv(os.path.join(output_folder, f'spike_{cluster}.csv'), index=False)

print(f"Todos los archivos se han guardado en la carpeta: {output_folder}")