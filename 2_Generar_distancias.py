import numpy as np
import pandas as pd
import time

start_time = time.time()

# Leer los datos del CSV, omitiendo la columna de nombre del filtro
spike_trains = pd.read_csv("spike_trains.csv").drop(columns=["filter"])
num_trains = spike_trains.shape[0]

def isi_distance(spike_train1, spike_train2, T0=0.0, T1=21.5, resolution=1000):
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

    t = np.linspace(T0, T1, resolution)
    isi1 = get_isi_profile(t, np.sort(spike_train1))
    isi2 = get_isi_profile(t, np.sort(spike_train2))

    with np.errstate(divide='ignore', invalid='ignore'):
        I = np.abs(isi1 - isi2) / np.maximum(isi1, isi2)
        I[np.isnan(I)] = 0
    return np.mean(I)

def spike_distance(spike_train1, spike_train2, T0=0.0, T1=21.5, resolution=1000):
    t_vals = np.linspace(T0, T1, resolution)
    S_total = 0
    count = 0

    s1 = np.sort(np.append(spike_train1, [T0, T1]))
    s2 = np.sort(np.append(spike_train2, [T0, T1]))

    for t in t_vals:
        try:
            t1_prev = s1[s1 <= t].max()
            t1_next = s1[s1 > t].min()
            t2_prev = s2[s2 <= t].max()
            t2_next = s2[s2 > t].min()
        except ValueError:
            continue

        ν1 = t1_next - t1_prev
        ν2 = t2_next - t2_prev

        if ν1 == 0 or ν2 == 0:
            continue

        delta_t1_P = np.min(np.abs(spike_train2 - t1_prev))
        delta_t1_F = np.min(np.abs(spike_train2 - t1_next))
        delta_t2_P = np.min(np.abs(spike_train1 - t2_prev))
        delta_t2_F = np.min(np.abs(spike_train1 - t2_next))

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

# Inicializar matrices
distance_matrix = np.zeros((num_trains, num_trains))
ISI_matrix = np.zeros((num_trains, num_trains))
SPIKE_matrix = np.zeros((num_trains, num_trains))

# Calcular distancias
for i in range(num_trains):
    for j in range(i + 1, num_trains):
        train_i = spike_trains.iloc[i].dropna().to_numpy()
        train_j = spike_trains.iloc[j].dropna().to_numpy()

        isi_dist = isi_distance(train_i, train_j)
        spike_dist = spike_distance(train_i, train_j)
        avg_dist = (isi_dist + spike_dist) / 2

        ISI_matrix[i, j] = ISI_matrix[j, i] = isi_dist
        SPIKE_matrix[i, j] = SPIKE_matrix[j, i] = spike_dist
        distance_matrix[i, j] = distance_matrix[j, i] = avg_dist

# Guardar resultados
pd.DataFrame(distance_matrix).to_csv("matriz_distancia.csv", index=False)
pd.DataFrame(ISI_matrix).to_csv("matriz_ISI.csv", index=False)
pd.DataFrame(SPIKE_matrix).to_csv("matriz_SPIKE.csv", index=False)

# Mostrar tiempo de ejecución
execution_time = time.time() - start_time
print(f"✅ Código ejecutado en {execution_time:.2f} segundos.")
