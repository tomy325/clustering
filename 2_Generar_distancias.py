# === 2_Generar_distancias.py optimizado ===
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import time

start_time = time.time()

# Leer los datos del CSV, omitiendo la columna de nombre del filtro
spike_trains_df = pd.read_csv("spike_trains.csv").drop(columns=["filter", "l", "v"])
num_trains = spike_trains_df.shape[0]

# Convertir cada fila en una lista de spikes (sin NaN)
spike_trains = [row.dropna().to_numpy() for _, row in spike_trains_df.iterrows()]

# ======================= FUNCIONES =======================

def get_isi_profile(t, spikes):
    isi_profile = np.zeros_like(t)
    for i, time in enumerate(t):
        prev = spikes[spikes <= time]
        next_ = spikes[spikes > time]
        if len(prev) == 0 or len(next_) == 0:
            isi_profile[i] = np.nan
            continue
        isi = next_[0] - prev[-1]
        isi_profile[i] = isi
    return isi_profile

def spike_distance(t_vals, s1, s2):
    S_total = 0
    count = 0

    s1_ext = np.sort(np.append(s1, [0.0, 21.5]))
    s2_ext = np.sort(np.append(s2, [0.0, 21.5]))

    for t in t_vals:
        try:
            t1_prev = s1_ext[s1_ext <= t].max()
            t1_next = s1_ext[s1_ext > t].min()
            t2_prev = s2_ext[s2_ext <= t].max()
            t2_next = s2_ext[s2_ext > t].min()
        except ValueError:
            continue

        ν1 = t1_next - t1_prev
        ν2 = t2_next - t2_prev

        if ν1 == 0 or ν2 == 0:
            continue

        delta_t1_P = np.min(np.abs(s2 - t1_prev))
        delta_t1_F = np.min(np.abs(s2 - t1_next))
        delta_t2_P = np.min(np.abs(s1 - t2_prev))
        delta_t2_F = np.min(np.abs(s1 - t2_next))

        x1_P = t - t1_prev
        x1_F = t1_next - t
        x2_P = t - t2_prev
        x2_F = t2_next - t

        S1 = (delta_t1_P * x1_F + delta_t1_F * x1_P) / ν1
        S2 = (delta_t2_P * x2_F + delta_t2_F * x2_P) / ν2

        num = S1 * ν2 + S2 * ν1
        denom = 0.5 * (ν1 + ν2) ** 2
        S = num / denom
        S_total += S
        count += 1

    return S_total / count if count > 0 else np.nan

# ======================= PRECOMPUTO =======================

T0, T1 = 0.0, 21.5
resolution = 1000
t_vals = np.linspace(T0, T1, resolution)

isi_profiles = [get_isi_profile(t_vals, spikes) for spikes in spike_trains]

# ======================= DISTANCIA ENTRE PARES =======================

def compute_distances(i, j):
    isi1 = isi_profiles[i]
    isi2 = isi_profiles[j]

    with np.errstate(divide='ignore', invalid='ignore'):
        I = np.abs(isi1 - isi2) / np.maximum(isi1, isi2)
        I[np.isnan(I)] = 0
    isi_dist = np.mean(I)

    spike_dist = spike_distance(t_vals, spike_trains[i], spike_trains[j])
    avg_dist = (isi_dist + spike_dist) / 2

    return i, j, isi_dist, spike_dist, avg_dist

# Ejecutar en paralelo
resultados = Parallel(n_jobs=-1, prefer="threads")(
    delayed(compute_distances)(i, j)
    for i in range(num_trains) for j in range(i + 1, num_trains)
)

# Inicializar matrices
ISI_matrix = np.zeros((num_trains, num_trains))
SPIKE_matrix = np.zeros((num_trains, num_trains))
distance_matrix = np.zeros((num_trains, num_trains))

# Rellenar matrices simétricas
for i, j, isi, spike, avg in resultados:
    ISI_matrix[i, j] = ISI_matrix[j, i] = isi
    SPIKE_matrix[i, j] = SPIKE_matrix[j, i] = spike
    distance_matrix[i, j] = distance_matrix[j, i] = avg

# ======================= GUARDAR =======================

pd.DataFrame(distance_matrix).to_csv("matriz_distancia.csv", index=False)
pd.DataFrame(ISI_matrix).to_csv("matriz_ISI.csv", index=False)
pd.DataFrame(SPIKE_matrix).to_csv("matriz_SPIKE.csv", index=False)

print(f"✅ Generar distancias isi y spike tardo {time.time() - start_time:.2f} segundos.")
