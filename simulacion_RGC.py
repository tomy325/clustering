# Librerías

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

r_min = 0.5
r_max = 100
c = 4

# Función r(x) #
def r_function(t, min=0.5, max=100, c=4): 
    r = ((2 * max - min) / (1 + np.exp(-c * (t - 1)))) + min
    return r

# Función de intensidad/hazard/riesgo/tasa
def filter_result(t):
    h = np.sin(2*t)*np.exp(-t**2/4)
    return h

#densidad de probabilidad gaussiana
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
        return np.sin(np.pi*t**2)
    if(t>12.5 and t<=14.5):
        return 0
    if(t>14.5 and t<=19.5):
        return 0.2*t*np.sin(3*np.pi*t)
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

    # ON fast sustained
p_on=1
l_fast= 0.4
v_sustained= 1.2

# OF fast sustained
p_of=-1
l_fast= 0.4
v_sustained= 1.2

# ON slow sustained
p_on=1
l_slow= 1
v_sustained= 1.2

# OF slow sustained\
p_of=-1
l_slow= 1
v_sustained= 1.2

# ON fast transient
p_on=1
l_fast= 0.4
v_transient= 0.65

# OF fast transient
p_of=-1
l_fast= 0.4
v_transient= 0.65

# ON slow transient
p_on=1
l_slow= 1
v_transient= 0.65

# OF slow transient
p_of=-1
l_slow= 1
v_transient= 0.65


# Crear un vector de tiempo limitado entre 0 y 21.5
t = np.linspace(0, 21.5, 1000)  # Aumentar el número de puntos para mayor precisión

# Evaluar las funciones gauss y estimulo en ese dominio
gauss_values = gauss(p_on,t, 0, l_fast, v_sustained)  # Parámetros arbitrarios para la gaussiana
estimulo_values = np.array([estimulo(i) for i in t])


# Realizar la convolución manual
response = linear_response(gauss_values, estimulo_values)

# Normalizar la respuesta entre -1 y 1
response_min = np.min(response)
response_max = np.max(response)

normalized_response = (response - response_min) / (response_max - response_min) * 2 - 1

# Graficar los resultados normalizados
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.plot(t, gauss_values, label='Gaussiana')
plt.title('Filtro')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, estimulo_values, label='Estimulo', color='orange')
plt.title(' Estimulo')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, normalized_response, label='Convolución Normalizada', color='green')
plt.title('linear response normalizada')
plt.grid(True)

plt.tight_layout()
plt.show()