import numpy as np
import pywt
import pandas as pd
from itertools import combinations
import os
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURACIÓN ===
wavelet_name = 'coif1'
level = 1  # Puedes parametrizarlo si deseas

# === CARGA DE DATOS ===
df = pd.read_csv("spike_trains.csv")
spike_times = df.drop(columns='filter', errors='ignore').values
num_trains = len(spike_times)

# === FUNCIONES ===
def spike_train_to_vector(spikes, length=2150):
    spikes = spikes[~np.isnan(spikes)]
    indices = np.clip((spikes * 100).astype(int), 0, length - 1)
    vec = np.zeros(length)
    vec[indices] = 1
    return vec

# === CALCULAR COEFICIENTES cA Y cD PARA CADA SPIKE TRAIN ===
phi_psi_list = []
for row in spike_times:
    vec = spike_train_to_vector(row)
    coeffs = pywt.wavedec(vec, wavelet=wavelet_name, level=level)
    cA, cD = coeffs[0], coeffs[1]
    phi_psi_list.append((cA, cD))

# === MATRIZ DE DISTANCIA USANDO cA Y cD ===
dist_matrix = np.zeros((num_trains, num_trains))

for i, j in combinations(range(num_trains), 2):
    phi_i, psi_i = phi_psi_list[i]
    phi_j, psi_j = phi_psi_list[j]

    # Padding si tamaños distintos
    max_len_phi = max(len(phi_i), len(phi_j))
    max_len_psi = max(len(psi_i), len(psi_j))
    phi_i_padded = np.pad(phi_i, (0, max_len_phi - len(phi_i)))
    phi_j_padded = np.pad(phi_j, (0, max_len_phi - len(phi_j)))
    psi_i_padded = np.pad(psi_i, (0, max_len_psi - len(psi_i)))
    psi_j_padded = np.pad(psi_j, (0, max_len_psi - len(psi_j)))

    delta_phi = phi_i_padded - phi_j_padded
    delta_psi = psi_i_padded - psi_j_padded
    distance = np.sum(delta_phi ** 2) + np.sum(delta_psi ** 2)

    dist_matrix[i, j] = distance
    dist_matrix[j, i] = distance

# === GUARDAR MATRIZ ===
df_dist = pd.DataFrame(dist_matrix)
df_dist.to_csv("wavalet_matriz.csv", index=False)
print(f"✅ Matriz de distancia wavelet guardada como 'wavalet_matriz.csv'")

