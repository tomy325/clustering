# Librerías

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Ajustar parametros para r(x)
r_min = 0.5
r_max = 100
c = 4



# Parámetros para los ensayos (Spikes)
num_trials = 100  # Número de ensayos
dt = 0.001       # Intervalo de tiempo (1 ms)




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



#Definir funciones
# Función r(x) #
def r_function(t, min=0.5, max=100, c=4): 
    r = ((2 * max - min) / (1 + np.exp(-c * (t - 1)))) + min
    return r

# Función de intensidad/hazard/riesgo/tasa
def filter_result(t):
    h = np.sin(2*t)*np.exp(-t**2/4)
    return h

#linear filter
def gauss(p,t,mu,sigma,v):
    pdf = norm.pdf(t, mu, sigma/2)
    kate=p*pdf*np.sin(2*np.pi*(t/sigma)**v)
    return kate

def estimulo(t):
    if(t<=1.5):
        return -1
    if(t>1.5 and t<=3.5):
        return 1
    if(t>3.5 and t<=5.5):
        return -1
    if(t>5.5 and t<=7.5):
        return 0
    if(t>7.5 and t<=12.5):
        return np.sin(np.pi*(t-7.5)**2)
    if(t>12.5 and t<=14.5):
        return 0
    if(t>14.5 and t<=19.5):
        return 0.2*(t-14.5)*np.sin(3*np.pi*(t-14.5))
    if(t>19.5 and t<=21.5):
        return 0

def linear_response(f, g):
    N = len(f)
    response = np.zeros(N)  # Crear un vector de ceros del mismo tamaño
    for n in range(N):
        # Calcular la suma de convolución para cada desplazamiento n
        for k in range(N):
            if n - k >= 0:
                response[n] += f[k] * g[n - k]
    return response




# Crear un vector de tiempo limitado entre 0 y 21.5
t = np.linspace(0, 21.5, 1000)  # Aumentar el número de puntos para mayor precisión


# Seleccionar el filtro deseado
selected_filter = 'OF_fast_sustained'  # Cambia esta clave para seleccionar otro filtro

# Obtener los parámetros correspondientes del diccionario
params = filters_params[selected_filter]
p = params['p']
l = params['l']
v = params['v']

# Evaluar las funciones gauss y estimulo en ese dominio
gauss_values = gauss(p,t, 0, l, v)  # Parámetros arbitrarios para la gaussiana
estimulo_values = np.array([estimulo(i) for i in t])


# Realizar la convolución manual
response = linear_response(gauss_values, estimulo_values)


# Normalizar la respuesta entre -1 y 1
response_min = np.min(response)
response_max = np.max(response)

normalized_response = (response - response_min) / (response_max - response_min) * 2 - 1

rate = np.array([r_function(j) for j in normalized_response])



# Crear matriz para almacenar los spikes de cada ensayo
spike_trains = np.zeros((num_trials, len(t)))

# Simular tren de spikes para cada ensayo
for trial in range(num_trials):
    spike_trains[trial] = np.random.rand(len(t)) < rate * dt






# --- Graficar los resultados ---

plt.figure(figsize=(10, 10))  

# Filtro
plt.subplot(5, 1, 1)
plt.plot(t, gauss_values, label='Gaussiana')
plt.title('Filtro')
plt.grid(True)

# Estimulo
plt.subplot(5, 1, 2)
plt.plot(t, estimulo_values, label='Estimulo', color='orange')
plt.title('Estimulo')
plt.grid(True)

# Convolución Normalizada
plt.subplot(5, 1, 3)
plt.plot(t, normalized_response, label='Convolución Normalizada', color='green')
plt.title('Linear response normalizada')
plt.grid(True)

# Rate (Tasa de disparo)
plt.subplot(5, 1, 4)
plt.plot(t, rate, label='Rate (Hz)', color='blue')
plt.title('Rate (Hz)')
plt.grid(True)

# Raster Plot con puntos para visualizar los trenes de spikes
plt.subplot(5, 1, 5)
for trial in range(num_trials):
    spike_times = t[spike_trains[trial] == 1]
    plt.scatter(spike_times, np.ones_like(spike_times) * trial, color='black', s=10)  

plt.xlabel('Time (s)')
plt.ylabel('Trial')
plt.title('spikes')
plt.ylim([-1, num_trials])
plt.grid(True)

plt.tight_layout()
plt.show()