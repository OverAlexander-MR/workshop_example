import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Definición de la red neuronal convolucional
class ConvRedNeuronal(nn.Module):
    def __init__(self):
        super(ConvRedNeuronal, self).__init__()
        # Capas convolucionales para extraer características
        self.caracteristicas = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Capas completamente conectadas para clasificación
        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),  # 10 clases para ejemplo con CIFAR-10
        )

    def forward(self, x):
        x = self.caracteristicas(x)
        x = self.clasificador(x)
        return x

    def extraer_caracteristicas(self, x):
        """Método para extraer características de una imagen"""
        return self.caracteristicas(x)


# Función para cargar y preparar los datos
def cargar_datos(batch_size=32):
    # Transformaciones para normalizar las imágenes
    transformaciones = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Carga el conjunto de datos CIFAR-10
    conjunto_entrenamiento = torchvision.datasets.CIFAR10(
        root="./datos", train=True, download=True, transform=transformaciones
    )
    conjunto_prueba = torchvision.datasets.CIFAR10(
        root="./datos", train=False, download=True, transform=transformaciones
    )

    # Crea los dataloaders
    cargador_entrenamiento = DataLoader(
        conjunto_entrenamiento, batch_size=batch_size, shuffle=True
    )
    cargador_prueba = DataLoader(conjunto_prueba, batch_size=batch_size, shuffle=False)

    return cargador_entrenamiento, cargador_prueba


# Función de entrenamiento
def entrenar_modelo(modelo, cargador_entrenamiento, epochs=5, learning_rate=0.001):
    # Definir función de pérdida y optimizador
    criterio = nn.CrossEntropyLoss()
    optimizador = optim.Adam(modelo.parameters(), lr=learning_rate)

    # Dispositivo para entrenamiento (GPU si está disponible)
    dispositivo = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo)

    # Historial de pérdida para visualización
    historial_perdida = []

    print(f"Entrenando en: {dispositivo}")

    # Ciclo de entrenamiento
    for epoca in range(epochs):
        perdida_total = 0.0

        for i, datos in enumerate(cargador_entrenamiento):
            # Obtener las imágenes y etiquetas
            imagenes, etiquetas = datos
            imagenes, etiquetas = imagenes.to(dispositivo), etiquetas.to(dispositivo)

            # Poner a cero los gradientes
            optimizador.zero_grad()

            # Forward pass
            salidas = modelo(imagenes)

            # Calcular pérdida
            perdida = criterio(salidas, etiquetas)

            # Backward pass y optimización
            perdida.backward()
            optimizador.step()

            # Estadísticas
            perdida_total += perdida.item()

            if i % 100 == 99:
                print(
                    f"[Época {epoca + 1}, Batch {i + 1}] pérdida: {perdida_total / 100:.3f}"
                )
                historial_perdida.append(perdida_total / 100)
                perdida_total = 0.0

        print(f"Época {epoca + 1} completada")

    print("¡Entrenamiento completado!")

    # Graficar la pérdida
    plt.figure(figsize=(10, 5))
    plt.plot(historial_perdida)
    plt.title("Pérdida durante el entrenamiento")
    plt.xlabel("Iteraciones (x100)")
    plt.ylabel("Pérdida")
    plt.show()

    return modelo


# Función para extraer características de una imagen
def extraer_caracteristicas(modelo, imagen, dispositivo="cpu"):
    modelo.eval()
    modelo = modelo.to(dispositivo)
    imagen = imagen.to(dispositivo)

    with torch.no_grad():
        caracteristicas = modelo.extraer_caracteristicas(imagen)

    return caracteristicas


# Función principal para demostrar el uso
def main():
    # Crear modelo
    modelo = ConvRedNeuronal()

    # Cargar datos
    cargador_entrenamiento, cargador_prueba = cargar_datos()

    # Entrenar modelo
    print("Iniciando entrenamiento...")
    modelo = entrenar_modelo(modelo, cargador_entrenamiento, epochs=2)

    # Guardar modelo entrenado
    torch.save(modelo.state_dict(), "modelo_extractor_caracteristicas.pth")
    print("Modelo guardado como 'modelo_extractor_caracteristicas.pth'")

    # Ejemplo: extraer características de una imagen
    # Obtener una imagen de muestra del conjunto de prueba
    imagenes_ejemplo, _ = next(iter(cargador_prueba))
    imagen_muestra = imagenes_ejemplo[0:1]  # Seleccionar una sola imagen

    # Extraer características
    caracteristicas = extraer_caracteristicas(modelo, imagen_muestra)

    print(f"Forma de las características extraídas: {caracteristicas.shape}")

    # Visualizar la imagen original
    img = imagen_muestra[0].cpu().numpy().transpose((1, 2, 0))
    img = img * 0.5 + 0.5  # Desnormalizar
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Imagen de entrada")
    plt.show()

    # Visualizar algunas de las características extraídas
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        fila, col = i // 4, i % 4
        mapa_caract = caracteristicas[0, i].cpu().numpy()
        axs[fila, col].imshow(mapa_caract, cmap="viridis")
        axs[fila, col].set_title(f"Filtro {i+1}")
        axs[fila, col].axis("off")
    plt.suptitle("Mapas de características extraídas")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
