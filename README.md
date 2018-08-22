# RBFN
El principal motivo de que suba mi trabajo a la plataforma es tener un respaldo de éste ya que le invertí un buen tiempo y
si es de ayuda para alguien más mejor.
-------------------------------------------------------------------------------------------------------------------------------
Se considera que como conocimientos previos para el análisis de esta implementación en python se necesita saber al menos qué es
cada cosa:

  - Clustering
  - Machin Learning
  - Básico de programación en Python (La documentación de python y sus librerias es muy amigable)
  - ¿Qué es un dataset?
  - Algoritmo kmeans

-------------------------------------------------------------------------------------------------------------------------------

Radial Basis Function Network (RBFN) en Python

Autor: Luis Angel Torres Grimaldo, Estudiante de Ingeniería en Tecnologías de la Información.
Ubicación: Universidad Politécnica de Victoria. Ciudad Victoria, Tamaulipas - México.

Descripción:
  Es un proyecto elaborado para la materia de Minería de Datos. Consta de la implementación de una red neuronal con objetivo
  de clasificación, la cual consta de 3 capas:
  
    - Capa entrada
    - Capa oculta
    - Capa de salida
    
  Se podría decir que esta implementación es una implementación de un algoritmo híbrido ya que primero se utiliza el kmeans
  para definir los k-centroides necesarios para la RBFN.
  
  En particular esta implementación trabaja con 4 datasets, todo de forma sistemática y para cada dataset se ejecuta con los
  valores de k=3 hasta k=8, en otras palabras para cada dataset se implementa la RBFN 6 veces con distintos valores de k.
  
