# Librerías

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Parámetros 
lambda_rate = 6.01  # tasa de llegada (eventos por unidad de tiempo)
total_time = 10  # tiempo total para la simulación
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
def gauss(t,mu,sigma,v):
    pdf = norm.pdf(t, mu, sigma/2)
    kate=pdf*np.sin(2*np.pi*(t/sigma)**v)
    return kate

def estimulo(t):
    if(t<=1.5):
        return 0
    if(t>1.5 and t<=3.5):
        return 1
    if(t>3.5 and t<=5.5):
        return 0
    if(t>5.5 and t<=7.5):
        return 0.5
    if(t>7.5 and t<=12.5):
        return np.sin(np.pi*t**2)
    if(t>12.5 and t<=14.5):
        return 0.5
    if(t>14.5 and t<=19.5):
        return 0.2*t*np.sin(3*np.pi*t)
    if(t>19.5 and t<=21.5):
        return 0.5
    


# Crear una figura con tres subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True, sharey=True)

# Primer gráfico: Proceso de Poisson simple
event_times = []
current_time = 0
while current_time < total_time:
    time_until_next_event = np.random.exponential(1 / lambda_rate)
    current_time += time_until_next_event
    if current_time < total_time:
        event_times.append(current_time)

axs[0].step([0] + event_times, range(len(event_times) + 1), where='post', label='Proceso de Poisson')
axs[0].scatter(event_times, range(1, len(event_times) + 1), color='red', zorder=5, label='Eventos')
axs[0].set_title('Proceso de Poisson Simple')
axs[0].set_ylabel('Número acumulado de eventos')
axs[0].grid(True)
axs[0].legend()

# Segundo gráfico: Proceso de Poisson con probabilidad variable
event_times = []
current_time = 0
while current_time < total_time:
    h = r_function(filter_result(current_time))
    time_until_next_event = np.random.exponential(1 / lambda_rate)
    current_time += time_until_next_event
    if (current_time < total_time and np.random.rand() < h / lambda_rate):
        event_times.append(current_time)

axs[1].step([0] + event_times, range(len(event_times) + 1), where='post', label='Proceso de Poisson')
axs[1].scatter(event_times, range(1, len(event_times) + 1), color='red', zorder=5, label='Eventos')
axs[1].set_title('Proceso de Poisson no homogeneo ')
axs[1].set_ylabel('Número acumulado de eventos')
axs[1].grid(True)
axs[1].legend()

# Tercer gráfico: Proceso de Poisson con función r(x) variable


event_times = []
current_time = 0
#lambda_rate tiene que ser más grande que el máximo valor posible de r_function 
lambda_rate = 200 #en caso que r_max = 100
while current_time < total_time:
    h = r_function(filter_result(current_time))
    time_until_next_event = np.random.exponential(1 / lambda_rate)
    current_time += time_until_next_event
    if (current_time < total_time and np.random.rand() < h / lambda_rate):
        event_times.append(current_time)

axs[2].step([0] + event_times, range(len(event_times) + 1), where='post', label='Proceso de Poisson')
axs[2].scatter(event_times, range(1, len(event_times) + 1), color='red', zorder=5, label='Eventos')
axs[2].set_title('Proceso de Poisson con Función r(x)')
axs[2].set_xlabel('Tiempo')
axs[2].set_ylabel('Número acumulado de eventos')
axs[2].grid(True)
axs[2].legend()

# Ajustar el layout y mostrar el gráfico
plt.tight_layout()
plt.show()



##################################################

#ON and OF
p_on=1
p_of=-1

# Fast and slow
l_fast= 0.4
l_slow= 1

# Transient and sustained
v_transient= 0.65
v_sustained= 1.2


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

