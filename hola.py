import os
import time

# Carita feliz en ASCII
carita = ":-)"

# Ancho del movimiento
ancho = 20

# Bucle para mover la carita
for i in range(ancho):
    # Limpia la pantalla
    os.system("cls" if os.name == "nt" else "clear")

    # Dibuja la carita en la posición actual
    print(" " * i + carita)

    # Pausa para que el movimiento sea visible
    time.sleep(0.1)
