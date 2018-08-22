# RBFN
El principal motivo de que suba mi trabajo a la plataforma es tener un respaldo de éste ya que le invertí un buen tiempo y
si es de ayuda para alguien más mejor. Además, hay un PDF dónde se describe de forma "no matemática" el algoritmo o la gran
mayoría de la implementación. Para ejecutar el programa en tu equipo solo hay que descargar el repositorio y cambiar la dirección de la variable "path" que se encuentra en el main. La ruta que se debe cambiar es la ruta donde se encuentran los dataset.
-------------------------------------------------------------------------------------------------------------------------------
Se considera que como conocimientos previos para el análisis de esta implementación en python se necesita saber al menos qué es
cada cosa:

  - Clustering
  - Machin Learning
  - Básico de programación en Python (La documentación de python y sus librerias es muy amigable)
  - ¿Qué es un dataset?
  - Algoritmo kmeans
  - Coeficiente de Relación de Matthews

-------------------------------------------------------------------------------------------------------------------------------

Radial Basis Function Network (RBFN) en Python

Autor: Luis Angel Torres Grimaldo, Estudiante de Ingeniería en Tecnologías de la Información.
Ubicación: Universidad Politécnica de Victoria. Ciudad Victoria, Tamaulipas - México.

Descripción:
  Es un proyecto elaborado para la materia de Minería de Datos. Consta de la implementación de una RBFN como alternativa de perceptrón multicapa, la cual consta de 3 capas:
  
    - Capa entrada
    - Capa oculta
    - Capa de salida
    
  Se podría decir que esta implementación es una implementación de un algoritmo híbrido ya que primero se utiliza el kmeans        (aprendizaje no supervizado) para definir los k-centroides necesarios para la RBFN (aprendizaje supervizado).
  
  En particular esta implementación trabaja con 4 datasets, todo de forma sistemática y para cada dataset se ejecuta con los
  valores de k=3 hasta k=8, en otras palabras para cada dataset se implementa la RBFN 6 veces con distintos valores de k.
  
  Ya que no se contó con el tiempo suficiente de implementar, por cuenta propia de un servidor, el algoritmo de Coeficiente de Relación de Matthews (CRM) multiclase se utilizó como alternativa la implementación de la librería de sklearn. El CRM multiclase sirve para medir la genericidad de las etiquetas que genera el clasificador.

--------------------------------------------------------------------------------------------------------------------------------

  Lo que finalmente genera la ejecución de ésta implementación son gráficas que representan el promedio del CRM para cada valor de K, para cada dataset. Para mayor comprensión de todo ésto se recomienda leer el reporte del proyecto.  
  
