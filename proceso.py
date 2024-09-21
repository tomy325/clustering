# Librerías
import numpy as np
import matplotlib.pyplot as plt

# Parámetros 
lambda_rate = 5  # tasa de llegada (eventos por unidad de tiempo)
total_time = 10  # tiempo total para la simulación
event_times = []
current_time = 0
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
axs[0].set_title('P1')
axs[0].set_ylabel('Número acumulado de eventos')
axs[0].grid(True)
axs[0].legend()

# Segundo gráfico: Proceso de Poisson con probabilidad variable
event_times = []
current_time = 0
while current_time < total_time:
    time_until_next_event = np.random.exponential(1 / lambda_rate)
    current_time += time_until_next_event
    if (current_time < total_time):
        event_times.append(current_time)

axs[1].step([0] + event_times, range(len(event_times) + 1), where='post', label='Proceso de Poisson')
axs[1].scatter(event_times, range(1, len(event_times) + 1), color='red', zorder=5, label='Eventos')
axs[1].set_title('P2 ')
axs[1].set_ylabel('Número acumulado de eventos')
axs[1].grid(True)
axs[1].legend()

# Tercer gráfico: Proceso de Poisson con función r(x) variable


event_times = []
current_time = 0
while current_time < total_time:
    time_until_next_event = np.random.exponential(1 / lambda_rate)
    current_time += time_until_next_event
    if (current_time < total_time):
        event_times.append(current_time)

axs[2].step([0] + event_times, range(len(event_times) + 1), where='post', label='Proceso de Poisson')
axs[2].scatter(event_times, range(1, len(event_times) + 1), color='red', zorder=5, label='Eventos')
axs[2].set_title('P3')
axs[2].set_xlabel('Tiempo')
axs[2].set_ylabel('Número acumulado de eventos')
axs[2].grid(True)
axs[2].legend()

# Ajustar el layout y mostrar el gráfico
plt.tight_layout()
plt.show()
