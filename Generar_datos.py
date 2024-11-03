import numpy as np
import pandas as pd  # Para guardar en CSV
from scipy.stats import norm

# Ajustar parámetros para r(x)
r_min = 0.5
r_max = 100
c = 4
lambda_rate = 200

# Parámetros para los ensayos (Spikes)
num_trials_per_filter = 5  # Número de ensayos por filtro

# Definir las combinaciones posibles de parámetros en un diccionario
filters_params = {
    'ON_fast_sustained': {'p': 1, 'l': 0.4, 'v': 1.2},
    'OF_fast_sustained': {'p': -1, 'l': 0.4, 'v': 1.2},
    'ON_slow_sustained': {'p': 1, 'l': 1, 'v': 1.2},
    'OF_slow_sustained': {'p': -1, 'l': 1, 'v': 1.2},
    'ON_fast_transient': {'p': 1, 'l': 0.4, 'v': 0.65},
    'OF_fast_transient': {'p': -1, 'l': 0.4, 'v': 0.65},
    'ON_slow_transient': {'p': 1, 'l': 1, 'v': 0.65},
    'OF_slow_transient': {'p': -1, 'l': 1, 'v': 0.65}
}

# Función r(x)
def r_function(t, min=0.5, max=100, c=4): 
    return ((2 * max - min) / (1 + np.exp(-c * (t - 1)))) + min

# Función filtro gaussiano
def gauss(p, t, mu, sigma, v):
    pdf = norm.pdf(t, mu, sigma / 2)
    return p * pdf * np.sin(2 * np.pi * (t / sigma) ** v)

# Función estímulo
def estimulo(t):
    if t <= 1.5:
        return -1
    elif t > 1.5 and t <= 3.5:
        return 1
    elif t > 3.5 and t <= 5.5:
        return -1
    elif t > 5.5 and t <= 7.5:
        return 0
    elif t > 7.5 and t <= 12.5:
        return np.sin(np.pi * (t - 7.5) ** 2)
    elif t > 12.5 and t <= 14.5:
        return 0
    elif t > 14.5 and t <= 19.5:
        return 0.2 * (t - 14.5) * np.sin(3 * np.pi * (t - 14.5))
    elif t > 19.5 and t <= 21.5:
        return 0

# Convolución lineal
def linear_response(f, g):
    N = len(f)
    response = np.zeros(N)
    for n in range(N):
        for k in range(N):
            if n - k >= 0:
                response[n] += f[k] * g[n - k]
    return response

# Crear un vector de tiempo entre 0 y 21.5
t = np.linspace(0, 21.5, 1000)

# Lista para almacenar los resultados y etiquetas
spike_data = []

# Iterar sobre cada filtro en el diccionario
for filter_name, params in filters_params.items():
    p = params['p']
    l = params['l']
    v = params['v']

    # Evaluar las funciones gauss y estimulo en ese dominio
    gauss_values = gauss(p, t, 0, l, v)
    estimulo_values = np.array([estimulo(i) for i in t])

    # Realizar la convolución manual
    response = linear_response(gauss_values, estimulo_values)

    # Normalizar la respuesta entre -1 y 1
    response_min = np.min(response)
    response_max = np.max(response)
    normalized_response = (response - response_min) / (response_max - response_min) * 2 - 1

    # Calcular el rate basado en la convolución normalizada
    rate = np.array([r_function(j) for j in normalized_response])

    # Simular tren de spikes para cada ensayo
    for trial in range(num_trials_per_filter):
        spike_times = [0]  # Iniciar la lista de tiempos con 0
        for i in range(len(t)):
            if np.random.rand() < rate[i] / lambda_rate:
                spike_times.append(t[i])  # Almacenar el tiempo en lugar de la posición
        spike_times.append(21.5)  # Asegurar que el tiempo 21.5 esté incluido
        spike_data.append(spike_times)  # Añadir los tiempos de este ensayo

# Convertir a DataFrame y guardar en CSV sin la columna de `trial`
spike_df = pd.DataFrame(spike_data)
spike_df.to_csv("spike_trains.csv", index=False)
