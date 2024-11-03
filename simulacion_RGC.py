import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Ajustar parámetros para r(x)
r_min = 0.5
r_max = 100
c = 4
lambda_rate = 200  # 200 en caso de que rmax sea igual a 100

# Parámetros para los ensayos (Spikes)
num_trials = 2  # Número de ensayos

# Definir todas las combinaciones posibles de parámetros en un diccionario
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

# Definir funciones
def r_function(t, min=0.5, max=100, c=4): 
    r = ((2 * max - min) / (1 + np.exp(-c * (t - 1)))) + min
    return r

def filter_result(t):
    h = np.sin(2*t)*np.exp(-t**2/4)
    return h

def gauss(p, t, mu, sigma, v):
    pdf = norm.pdf(t, mu, sigma / 2)
    kate = p * pdf * np.sin(2 * np.pi * (t / sigma) ** v)
    return kate

def estimulo(t):
    if t <= 1.5:
        return -1
    elif 1.5 < t <= 3.5:
        return 1
    elif 3.5 < t <= 5.5:
        return -1
    elif 5.5 < t <= 7.5:
        return 0
    elif 7.5 < t <= 12.5:
        return np.sin(np.pi * (t - 7.5) ** 2)
    elif 12.5 < t <= 14.5:
        return 0
    elif 14.5 < t <= 19.5:
        return 0.2 * (t - 14.5) * np.sin(3 * np.pi * (t - 14.5))
    else:
        return 0

def linear_response(f, g):
    N = len(f)
    response = np.zeros(N)  
    for n in range(N):
        for k in range(N):
            if n - k >= 0:
                response[n] += f[k] * g[n - k]
    return response

# Crear un vector de tiempo
t = np.linspace(0, 21.5, 1000)

# Seleccionar el filtro deseado
selected_filter = 'OF_fast_sustained'
params = filters_params[selected_filter]
p = params['p']
l = params['l']
v = params['v']

# Evaluar las funciones gauss y estimulo en el dominio t
gauss_values = gauss(p, t, 0, l, v)
estimulo_values = np.array([estimulo(i) for i in t])

# Realizar la convolución manual
response = linear_response(gauss_values, estimulo_values)

# Normalizar la respuesta entre -1 y 1
response_min = np.min(response)
response_max = np.max(response)
normalized_response = (response - response_min) / (response_max - response_min) * 2 - 1

# Calcular la tasa de disparo usando r_function
rate = np.array([r_function(j) for j in normalized_response])

# Crear una lista para almacenar los tiempos de spikes de cada ensayo
spike_trains = []

# Simular tren de spikes para cada ensayo
for trial in range(num_trials):
    spike_times = []  # Lista para almacenar tiempos de spikes en este ensayo
    for i in range(len(t)):
        if np.random.rand() < rate[i] / lambda_rate:
            spike_times.append(t[i])  # Almacenar el tiempo en lugar de la posición
    spike_trains.append(spike_times)  # Añadir los tiempos de este ensayo

print(len(spike_trains[0]))
print(len(spike_trains[1]))
# --- Graficar los resultados ---
plt.figure(figsize=(10, 10))

# Filtro
plt.subplot(5, 1, 1)
plt.plot(t, gauss_values, label='Gaussiana')
plt.title('Filtro')
plt.grid(True)

# Estimulo
plt.subplot(5, 1, 2)
plt.plot(t, estimulo_values, label='Estimulo', color='red')
plt.title('Estimulo')
plt.grid(True)

# Convolución Normalizada
plt.subplot(5, 1, 3)
plt.plot(t, normalized_response, label='Convolución Normalizada', color='purple')
plt.title('Respuesta Lineal Normalizada')
plt.grid(True)

# Rate (Tasa de disparo)
plt.subplot(5, 1, 4)
plt.plot(t, rate, label='Rate (Hz)', color='green')
plt.title('Rate (Hz)')
plt.grid(True)

# Raster Plot con tiempos de spikes
plt.subplot(5, 1, 5)
for trial, spike_times in enumerate(spike_trains):
    plt.scatter(spike_times, np.ones_like(spike_times) * trial, color='black', s=10)

plt.xlabel('Tiempo (s)')
plt.ylabel('Ensayo')
plt.title('Spikes')
plt.ylim([-1, num_trials])
plt.grid(True)

plt.tight_layout()
plt.show()
