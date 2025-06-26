import pandas as pd
import numpy as np
import math
from collections import Counter
import random

datos_entrenamiento = None
X_train = None
y_train = None


def distancia_euclidiana(punto1, punto2):

    p1 = np.array(punto1, dtype=float)
    p2 = np.array(punto2, dtype=float)
    return math.sqrt(sum((p1 - p2) ** 2))


def distancia_manhattan(punto1, punto2):
    """
    Calcula la distancia manhattan entre dos puntos
    """
    p1 = np.array(punto1, dtype=float)
    p2 = np.array(punto2, dtype=float)
    return sum(np.abs(p1 - p2))


def cargar_archivo(nombre_archivo):

    try:

        print("\n¿Su archivo CSV tiene encabezados (nombres de columnas en la primera fila)?")
        print("1. Sí, tiene encabezados")
        print("2. No, no tiene encabezados")
        opcion_header = input("Seleccione (1 o 2): ")

        if opcion_header == "2":
            datos = pd.read_csv(nombre_archivo, header=None)
            print("Archivo cargado SIN encabezados")
        else:
            datos = pd.read_csv(nombre_archivo)
            print("Archivo cargado CON encabezados")

        print("\n" + "=" * 50)
        print("DATOS CARGADOS EXITOSAMENTE")
        print("=" * 50)
        print(f"Archivo: {nombre_archivo}")
        print(f"Dimensiones: {datos.shape}")
        print(f"Columnas disponibles: {list(datos.columns)}")
        print("\nPrimeras 5 filas:")
        print(datos.head())
        print("\nInformación del dataset:")
        print(datos.info())
        return datos
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None


def seleccionar_columnas(datos):
    global X_train, y_train

    print("\n" + "=" * 50)
    print("SELECCIÓN DE VARIABLES")
    print("=" * 50)

    columnas = list(datos.columns)
    print("Columnas disponibles:")
    for i, col in enumerate(columnas):
        print(f"{i + 1}. {col}")

    print("\nSeleccione las columnas para el vector de ENTRADA (features):")
    entrada_indices = input("Ingrese los números separados por comas (ej: 1,2,3): ")
    try:
        indices_entrada = [int(x.strip()) - 1 for x in entrada_indices.split(',')]
        columnas_entrada = [columnas[i] for i in indices_entrada]
        print(f"Columnas de entrada seleccionadas: {columnas_entrada}")
    except:
        print("Error en la selección. Usando las primeras columnas por defecto.")
        columnas_entrada = columnas[:-1]

    print("\nSeleccione la columna para el vector de SALIDA (target):")
    salida_index = input("Ingrese el número: ")
    try:
        indice_salida = int(salida_index.strip()) - 1
        columna_salida = columnas[indice_salida]
        print(f"Columna de salida seleccionada: {columna_salida}")
    except:
        print("Error en la selección. Usando la última columna por defecto.")
        columna_salida = columnas[-1]
    X_train = datos[columnas_entrada].values
    y_train = datos[columna_salida].values

    print(f"\nDimensiones del vector de entrada: {X_train.shape}")
    print(f"Dimensiones del vector de salida: {y_train.shape}")
    print(f"Clases únicas: {np.unique(y_train)}")

    return X_train, y_train


def calcular_centroides(X, y):
    centroides = {}
    clases_unicas = np.unique(y)

    print("\n" + "=" * 50)
    print("CÁLCULO DE CENTROIDES (MÍNIMA DISTANCIA)")
    print("=" * 50)

    for clase in clases_unicas:
        indices_clase = np.where(y == clase)[0]
        puntos_clase = X[indices_clase]

        centroide = np.mean(puntos_clase, axis=0)
        centroides[clase] = centroide

        print(f"Clase {clase}: {len(puntos_clase)} muestras")
        print(f"Centroide: {centroide}")

    return centroides


def clasificar_minima_distancia(punto_test, centroides, tipo_distancia='euclidiana'):
    distancias = {}

    for clase, centroide in centroides.items():
        if tipo_distancia == 'euclidiana':
            dist = distancia_euclidiana(punto_test, centroide)
        else:
            dist = distancia_manhattan(punto_test, centroide)
        distancias[clase] = dist


    clase_predicha = min(distancias, key=distancias.get)
    return clase_predicha, distancias


def clasificar_knn(punto_test, X_train, y_train, k=3, tipo_distancia='euclidiana'):

    distancias = []

    print(f"DEBUG KNN - Punto a clasificar: {punto_test}")
    print(f"DEBUG KNN - Tipo distancia: {tipo_distancia}")
    print(f"DEBUG KNN - Total puntos entrenamiento: {len(X_train)}")

    for i, punto_entrenamiento in enumerate(X_train):
        if tipo_distancia == 'euclidiana':
            dist = distancia_euclidiana(punto_test, punto_entrenamiento)
        else:
            dist = distancia_manhattan(punto_test, punto_entrenamiento)

        distancias.append((dist, y_train[i], i))

        if i < 5:
            print(f"DEBUG KNN - Punto {i}: {punto_entrenamiento}, Clase: {y_train[i]}, Distancia: {dist}")

    print(f"DEBUG KNN - Primeras 10 distancias sin ordenar:")
    for j in range(min(10, len(distancias))):
        print(f"  {j}: dist={distancias[j][0]:.4f}, clase={distancias[j][1]}, idx={distancias[j][2]}")

    distancias.sort(key=lambda x: x[0])
    k_vecinos = distancias[:k]

    print(f"DEBUG KNN - Los {k} vecinos más cercanos DESPUÉS de ordenar:")
    for j, (dist, clase, idx) in enumerate(k_vecinos):
        print(f"  {j + 1}: idx={idx}, clase={clase}, distancia={dist:.4f}")

    # Contar votos
    votos = [vecino[1] for vecino in k_vecinos]
    contador = Counter(votos)
    clase_predicha = contador.most_common(1)[0][0]

    return clase_predicha, k_vecinos

def ingresar_punto_manual():
    print("\n" + "=" * 50)
    print("INGRESO MANUAL DE PUNTO")
    print("=" * 50)

    if X_train is None:
        print("Error: No hay datos cargados")
        return None

    n_features = X_train.shape[1]
    print(f"El vector debe tener {n_features} características")

    try:
        valores = input(f"Ingrese {n_features} valores separados por comas: ")
        punto = [float(x.strip()) for x in valores.split(',')]

        if len(punto) != n_features:
            print(f"Error: Se esperaban {n_features} valores, se recibieron {len(punto)}")
            return None

        return np.array(punto)
    except:
        print("Error al convertir los valores. Asegúrese de usar números.")
        return None


def evaluar_algoritmos():
    global X_train, y_train

    if X_train is None or y_train is None:
        print("Error: No hay datos cargados")
        return

    centroides = calcular_centroides(X_train, y_train)

    while True:
        print("\n" + "=" * 50)
        print("MENÚ DE CLASIFICACIÓN")
        print("=" * 50)
        print("1. Ingresar punto manualmente")
        print("2. Configurar parámetros")
        print("3. Volver al menú principal")
        opcion = input("\nSeleccione una opción: ")
        if opcion == '1':
            punto = ingresar_punto_manual()
            if punto is not None:
                clasificar_punto(punto, centroides)

        elif opcion == '2':
            configurar_parametros(centroides)

        elif opcion == '3':
            break
        else:
            print("Opción no válida")


def configurar_parametros(centroides):

    print("\n" + "=" * 50)
    print("CONFIGURACIÓN DE PARÁMETROS")
    print("=" * 50)

    # Tipo de distancia
    print("Seleccione tipo de distancia:")
    print("1. Euclidiana")
    print("2. Manhattan")
    tipo_dist = input("Opción (1 o 2): ")
    tipo_distancia = 'euclidiana' if tipo_dist == '1' else 'manhattan'

    try:
        k = int(input("Ingrese el valor de K para KNN: "))
        if k <= 0:
            k = 3
            print("Valor inválido. Usando K=3")
    except:
        k = 3
        print("Valor inválido. Usando K=3")

    punto = ingresar_punto_manual()
    if punto is not None:
        clasificar_punto(punto, centroides, tipo_distancia, k)


def clasificar_punto(punto, centroides, tipo_distancia='euclidiana', k=3):

    print("\n" + "=" * 50)
    print("RESULTADOS DE CLASIFICACIÓN")
    print("=" * 50)
    print(f"Punto a clasificar: {punto}")
    print(f"Tipo de distancia: {tipo_distancia}")

    print(f"\n--- CLASIFICADOR MÍNIMA DISTANCIA ---")
    clase_md, distancias_md = clasificar_minima_distancia(punto, centroides, tipo_distancia)
    print(f"Clase predicha: {clase_md}")
    print("Distancias a centroides:")
    for clase, dist in distancias_md.items():
        print(f"  Clase {clase}: {dist:.4f}")

    print(f"\n--- CLASIFICADOR K-NN (K={k}) ---")
    clase_knn, vecinos = clasificar_knn(punto, X_train, y_train, k, tipo_distancia)
    print(f"Clase predicha: {clase_knn}")
    print(f"Los {k} vecinos más cercanos:")
    for i, (dist, clase, idx) in enumerate(vecinos):
        print(f"  {i + 1}. Índice: {idx}, Clase: {clase}, Distancia: {dist:.4f}")


def mostrar_estadisticas():
    """
    Muestra estadísticas básicas de los datos cargados
    """
    global X_train, y_train

    if X_train is None:
        print("No hay datos cargados")
        return

    print("\n" + "=" * 50)
    print("ESTADÍSTICAS DE LOS DATOS")
    print("=" * 50)

    df_temp = pd.DataFrame(X_train)
    df_temp['target'] = y_train

    print("Estadísticas descriptivas de las features:")
    print(df_temp.describe())

    print(f"\nDistribución de clases:")
    clases, conteos = np.unique(y_train, return_counts=True)
    for clase, conteo in zip(clases, conteos):
        print(f"Clase {clase}: {conteo} muestras ({conteo / len(y_train) * 100:.1f}%)")


def main():
    """
    Función principal del programa
    """
    print("=" * 60)
    print("ALGORITMOS BASADOS EN DISTANCIA")
    print("K-NN y Mínima Distancia")
    print("=" * 60)

    while True:
        print("\n" + "=" * 50)
        print("MENÚ PRINCIPAL")
        print("=" * 50)
        print("1. Cargar archivo de datos")
        print("2. Mostrar estadísticas de datos")
        print("3. Clasificar puntos")
        print("4. Salir")

        opcion = input("\nSeleccione una opción: ")

        if opcion == '1':
            nombre_archivo = input("\nIngrese el nombre del archivo CSV: ")
            datos = cargar_archivo(nombre_archivo)
            if datos is not None:
                seleccionar_columnas(datos)

        elif opcion == '2':
            mostrar_estadisticas()

        elif opcion == '3':
            evaluar_algoritmos()

        elif opcion == '4':
            print("¡Hasta luego!")
            break

        else:
            print("Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()