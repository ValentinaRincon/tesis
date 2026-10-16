# Modelos de difusión generativa

Código desarrollado para mi tesis de pregrado en Física en la Universidad de los Andes.

El proyecto estudia la implementación y análisis de modelos de difusión generativa,
con énfasis en su relación con procesos estocásticos, la ecuación de Langevin y el
score de la distribución.

## Experimentos

Actualmente se han implementado:

- DDPM sobre MNIST.
- Análisis del score durante el proceso de generación.
- Distribución bimodal 2D con score analítico y aprendido.
- DDPM sobre CIFAR-10 en RGB.
- U-Net con self-attention para CIFAR-10.
- Comparación cualitativa del proceso reverse y de las muestras generadas.

## Tecnologías

- Python
- PyTorch
- NumPy
- Matplotlib
- torchvision
- CUDA

## Estructura

```text
.
├── Prueba_con_MNIST.py
├── Prueba_con_CIFAR10.ipynb
├── Prueba_con_CIFAR10_Attention.ipynb
├── Score en MNIST.ipynb
├── Score en 2D.ipynb
├── modelos/
└── imágenes/
