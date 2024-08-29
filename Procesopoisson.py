#Librerias
import numpy as np
import matplotlib.pyplot as plt



# Parámetros del proceso de Poisson
lambda_rate = 5  # tasa de llegada (eventos por unidad de tiempo)
total_time = 10  # tiempo total para la simulación

# Inicializar variables
event_times = []
current_time = 0
# Simular el proceso de Poisson
while current_time < total_time:
    time_until_next_event = -np.log(np.random.uniform()) / lambda_rate
    current_time += time_until_next_event
    if current_time < total_time:
        event_times.append(current_time)

# Graficar el proceso de Poisson
plt.figure(figsize=(10, 6))

# Gráfico de escalera (número acumulado de eventos)
plt.step([0] + event_times, range(len(event_times) + 1), where='post', label='Proceso de Poisson')
plt.scatter(event_times, range(1, len(event_times) + 1), color='red', zorder=5, label='Eventos')

# Personalización del gráfico
plt.title('Simulación de un Proceso de Poisson')
plt.xlabel('Tiempo')
plt.ylabel('Número acumulado de eventos')
plt.grid(True)
plt.legend()

# Mostrar el gráfico
plt.show()
