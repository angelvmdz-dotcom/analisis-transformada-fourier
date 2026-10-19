import numpy as np
import matplotlib.pyplot as plt

# 1. Parametros generales
fs = 1000
duracion = 1.0
t = np.arange(0, duracion, 1/fs)

# 2. Funcion para calcular la FFT
def espectro(x, fs):
    """Calcula frecuencias, magnitud y fase de la FFT."""
    X = np.fft.fft(x)
    frecuencias = np.fft.fftfreq(len(x), d=1/fs)
    mitad = len(x) // 2
    magnitud = 2 * np.abs(X[:mitad]) / len(x)
    fase = np.angle(X[:mitad])
    return frecuencias[:mitad], magnitud, fase

# 3. Generacion de señales
pulso = np.zeros_like(t)
pulso[(t >= 0.20) & (t < 0.40)] = 1.0
escalon = np.where(t >= 0.30, 1.0, 0.0)
senoidal = np.sin(2 * np.pi * 10 * t)

# 4. Graficas de tiempo, magnitud y fase
senales = [
    ("Pulso rectangular", pulso),
    ("Funcion escalon", escalon),
    ("Senal senoidal de 10 Hz", senoidal)
]
fig, ejes = plt.subplots(3, 3, figsize=(15, 10))
for fila, (nombre, x) in enumerate(senales):
    frecuencias, magnitud, fase = espectro(x, fs)

    ejes[fila, 0].plot(t, x)
    ejes[fila, 0].set_title(nombre)
    ejes[fila, 0].set_xlabel("Tiempo (s)")
    ejes[fila, 0].set_ylabel("Amplitud")
    ejes[fila, 0].grid(True)

    ejes[fila, 1].plot(frecuencias, magnitud)
    ejes[fila, 1].set_title("Magnitud del espectro")
    ejes[fila, 1].set_xlabel("Frecuencia (Hz)")
    ejes[fila, 1].set_ylabel("Magnitud")
    ejes[fila, 1].set_xlim(0, 100)
    ejes[fila, 1].grid(True)

    ejes[fila, 2].plot(frecuencias, fase)
    ejes[fila, 2].set_title("Fase del espectro")
    ejes[fila, 2].set_xlabel("Frecuencia (Hz)")
    ejes[fila, 2].set_ylabel("Fase (rad)")
    ejes[fila, 2].set_xlim(0, 100)
    ejes[fila, 2].grid(True)

plt.tight_layout()
plt.savefig("senales_y_espectros.png", dpi=160)
plt.show()

# 5. Propiedad de linealidad
x = np.sin(2 * np.pi * 10 * t)
y = np.cos(2 * np.pi * 20 * t)
a, b = 2.0, 0.5
izquierda = np.fft.fft(a*x + b*y)
derecha = a*np.fft.fft(x) + b*np.fft.fft(y)
error = np.max(np.abs(izquierda - derecha))
print("PROPIEDAD DE LINEALIDAD")
print("Error maximo:", error)

# 6. Desplazamiento temporal
desplazamiento = 100
x_desplazada = np.roll(x, desplazamiento)
X = np.fft.fft(x)
X_desplazada = np.fft.fft(x_desplazada)
error_magnitud = np.max(np.abs(np.abs(X) - np.abs(X_desplazada)))
print("\nDESPLAZAMIENTO TEMPORAL")
print("Diferencia de magnitud:", error_magnitud)

# 7. Escalamiento de frecuencia: senoidales de 10 y 20 Hz
x10 = np.sin(2 * np.pi * 10 * t)
x20 = np.sin(2 * np.pi * 20 * t)
fig, ejes = plt.subplots(2, 2, figsize=(12, 7))

ejes[0, 0].plot(t, x10)
ejes[0, 0].set_title("Senal de 10 Hz")
ejes[0, 0].set_xlim(0, 0.3)
ejes[0, 0].set_xlabel("Tiempo (s)")
ejes[0, 0].grid(True)

ejes[0, 1].plot(t, x20)
ejes[0, 1].set_title("Senal de 20 Hz")
ejes[0, 1].set_xlim(0, 0.3)
ejes[0, 1].set_xlabel("Tiempo (s)")
ejes[0, 1].grid(True)

for senal, ax, nombre in [
    (x10, ejes[1, 0], "Espectro de 10 Hz"),
    (x20, ejes[1, 1], "Espectro de 20 Hz")
]:
    frecuencias, magnitud, fase = espectro(senal, fs)
    ax.plot(frecuencias, magnitud)
    ax.set_title(nombre)
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Magnitud")
    ax.set_xlim(0, 50)
    ax.grid(True)

plt.tight_layout()
plt.savefig("escalamiento_frecuencia.png", dpi=160)
plt.show()

# 8. Resultados finales
print("\nANALISIS TERMINADO")
print("Se generaron las graficas de las señales.")
print("Revisa los archivos PNG en la carpeta donde ejecutaste el programa.")
