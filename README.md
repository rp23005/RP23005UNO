# rp23005uno

**rp23005uno** es una librería en Python diseñada para resolver sistemas de ecuaciones lineales y no lineales. Está pensada como una herramienta educativa para estudiantes o profesionales que quieran aplicar métodos numéricos clásicos con un enfoque práctico y accesible.

## Características

- Métodos para sistemas de ecuaciones **lineales**:
  - Crammer
  - Eliminación de Gauss
  - Gauss-Jordan
  - Descomposición LU
  - Métodos iterativos: Jacobi y Gauss-Seidel
- Método para ecuaciones **no lineales**:
  - Bisección

## Requisitos

- Python >= 3.10 
- numpy >= 1.24

Al instalar `RP23005UNO`, **`numpy`** se instalará automáticamente si no lo tienes ya en tu entorno.


## Instalación

Para instalar la librería desde PyPI:

```bash
pip install rp23005uno
```

## ¿Cómo importar el paquete?
## Forma 1, importa uno por uno:
``` python
from rp23005uno.lineales import crammer
from rp23005uno.lineales import bisection  
```
## Forma 2, puede importarlos a todos o los necesarios:
``` python
from rp23005uno import crammer, bisection
```

## Ejemplos del uso de cada uno de los métodos:

```python
#-----------------------------ELIMINACIÓN GAUSSANIANA/GAUSS ELIMINATION------------------------------------------
from rp23005uno import gauss_elimination

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]  
b = [8, -11, -3]  

solucion = gauss_elimination(A, b)
print("Solución por Gauss:", solucion)

#---------------------------------------GAUSS JORDAN--------------------------------------------------------------------------
from rp23005uno import gauss_jordan

A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]  
b = [8, -11, -3] 

solucion = gauss_jordan(A, b)
print("Solución por Gauss-Jordan:", solucion)

#-----------------------------GAUSS SEIDEL----------------------------------------------------------------------------------------
import numpy as np
from rp23005uno import gauss_seidel

A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]], dtype=float)
b = np.array([7.85, -19.3, 71.4], dtype=float)

solucion = gauss_seidel(A, b)
print(f"Solución por Gauss-Seidel: {solucion}")

#-----------------------------JACOBI-----------------------------------------------------------------------------------------------
import numpy as np
from rp23005uno import jacobi

A = np.array([[3, -0.1, -0.2], [0.1, 7, -0.3], [0.3, -0.2, 10]], dtype=float)
b = np.array([7.85, -19.3, 71.4], dtype=float) 

solucion = jacobi(A, b)
print(f"Solución por Jacobi: {solucion}")

#---------------------------------------CRAMMER-------------------------------------------------------------------------------------
from rp23005uno import crammer
# Ejemplo 1: Sistema 2x2
coeficientes_2x2 = [[2, 1], [1, -1]]
constantes_2x2 = [5, 1]
sol = crammer(coeficientes_2x2, constantes_2x2)
print(f"Solución por crammer (2x2): {[str(frac) for frac in sol]}") 

# Ejemplo 2: Sistema 3x3
coeficientes_3x3 = [[1, 2, 1], [3, 1, 1], [2, 3, -1]]
constantes_3x3 = [7, 5, 3]
sol = crammer(coeficientes_3x3, constantes_3x3)
print(f"Solución por crammer (3x3): {[str(frac) for frac in sol]}")  

# Ejemplo 3: Sistema sin solución única
coeficientes_singular = [[1, 2], [2, 4]]
constantes_singular = [5, 10]
sol = crammer(coeficientes_singular, constantes_singular)
print(f"Solución por crammer (sistema singular): {sol}") 

#-----------------------------DESCOMPOSICION LU/LU DECOMPOSITION-------------------------------------------------------------------
from rp23005uno import lu_solver
# Ejemplo 1: Sistema 2x2
A1 = [[2, 1], [1, -1]]
b1 = [5, 1]
sol1 = lu_solver(A1, b1)
print(f"Solución por LU (2x2): {sol1}") 

# Ejemplo 2: Sistema 3x3
A2 = [[1, 1, 1], [0, 2, 5], [2, 5, -1]]
b2 = [6, -4, 27]
sol2 = lu_solver(A2, b2)
print(f"Solución por LU (3x3): {sol2}") 

# Ejemplo 3: Sistema singular
A3 = [[1, 2], [2, 4]]
b3 = [5, 10]
sol3 = lu_solver(A3, b3)
print(f"Solución por LU (sistema singular): {sol3}") 

#-----------------------------BISECCION/BISECTION------------------------------------------------------------------------------------
from rp23005uno import bisection

funcion = lambda x: x**4 + 3*x**3 - 2

raiz, iteraciones = bisection(funcion,0,1) #intervalo
print(f"Raiz de funcion por biseccion es igual a {raiz:.6f}, encontrada en un total de {iteraciones} iteraciones.")
```