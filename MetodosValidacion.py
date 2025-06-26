import pandas as pd
import numpy as np
import math
import random
from collections import Counter

datos_cargados = None
X_data = None
y_data = None


def cargar_archivo(nombre_archivo):
    global datos_cargados

    try:
        print("¿Su archivo CSV tiene encabezados (nombres de columnas en la primera fila)?")
        print("1. Sí, tiene encabezados")
        print("2. No, no tiene encabezados")
        opcion_header = input("Seleccione (1 o 2): ")

        if opcion_header == "2":
            datos_cargados = pd.read_csv(nombre_archivo, header=None)
            print("Archivo cargado SIN encabezados")
        else:
            datos_cargados = pd.read_csv(nombre_archivo)
            print("Archivo cargado CON encabezados")

        print("DATOS CARGADOS EXITOSAMENTE")
        print(f"Archivo: {nombre_archivo}")
        print(f"Dimensiones: {datos_cargados.shape}")
        print(f"Columnas disponibles: {list(datos_cargados.columns)}")
        print("Primeras 5 filas:")
        print(datos_cargados.head())

        return datos_cargados
    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None


def seleccionar_columnas(datos):
    global X_data, y_data

    print("SELECCIÓN DE VARIABLES")

    columnas = list(datos.columns)
    print("Columnas disponibles:")
    for i, col in enumerate(columnas):
        print(f"{i + 1}. {col}")

    print("Seleccione las columnas para el vector de ENTRADA (features):")
    entrada_indices = input("Ingrese los números separados por comas (ej: 1,2,3): ")
    try:
        indices_entrada = [int(x.strip()) - 1 for x in entrada_indices.split(',')]
        columnas_entrada = [columnas[i] for i in indices_entrada]
        print(f"Columnas de entrada seleccionadas: {columnas_entrada}")
    except:
        print("Error en la selección. Usando las primeras columnas por defecto.")
        columnas_entrada = columnas[:-1]

    print("Seleccione la columna para el vector de SALIDA (target):")
    salida_index = input("Ingrese el número: ")
    try:
        indice_salida = int(salida_index.strip()) - 1
        columna_salida = columnas[indice_salida]
        print(f"Columna de salida seleccionada: {columna_salida}")
    except:
        print("Error en la selección. Usando la última columna por defecto.")
        columna_salida = columnas[-1]

    X_data = datos[columnas_entrada].values
    y_data = datos[columna_salida].values

    print(f"Dimensiones del vector de entrada: {X_data.shape}")
    print(f"Dimensiones del vector de salida: {y_data.shape}")
    print(f"Clases únicas: {np.unique(y_data)}")

    return X_data, y_data


def distancia_euclidiana(punto1, punto2):
    p1 = np.array(punto1, dtype=float)
    p2 = np.array(punto2, dtype=float)
    return math.sqrt(sum((p1 - p2) ** 2))


def calcular_centroides(X, y):
    centroides = {}
    clases_unicas = np.unique(y)

    for clase in clases_unicas:
        indices_clase = np.where(y == clase)[0]
        puntos_clase = X[indices_clase]
        centroide = np.mean(puntos_clase, axis=0)
        centroides[clase] = centroide

    return centroides


def clasificar_minima_distancia(punto_test, centroides):
    distancias = {}

    for clase, centroide in centroides.items():
        dist = distancia_euclidiana(punto_test, centroide)
        distancias[clase] = dist

    clase_predicha = min(distancias, key=distancias.get)
    return clase_predicha


def clasificar_knn(punto_test, X_train, y_train, k=3):
    distancias = []

    for i, punto_entrenamiento in enumerate(X_train):
        dist = distancia_euclidiana(punto_test, punto_entrenamiento)
        distancias.append((dist, y_train[i]))

    distancias.sort(key=lambda x: x[0])
    k_vecinos = distancias[:k]

    votos = [vecino[1] for vecino in k_vecinos]
    contador = Counter(votos)
    clase_predicha = contador.most_common(1)[0][0]

    return clase_predicha


def calcular_metricas(y_true, y_pred):
    total = len(y_true)
    correctos = sum(1 for i in range(total) if y_true[i] == y_pred[i])

    accuracy = (correctos / total) * 100
    error = 100 - accuracy

    return accuracy, error, correctos, total


def train_test_validation():
    if X_data is None or y_data is None:
        print("Error: No hay datos cargados")
        return

    print("VALIDACIÓN TRAIN AND TEST")

    try:
        porcentaje_train = float(input("Ingrese el porcentaje para entrenamiento (ej: 70): "))
        if porcentaje_train <= 0 or porcentaje_train >= 100:
            porcentaje_train = 70
            print("Valor inválido. Usando 70% por defecto.")
    except:
        porcentaje_train = 70
        print("Valor inválido. Usando 70% por defecto.")

    n_total = len(X_data)
    n_train = int(n_total * porcentaje_train / 100)

    indices = list(range(n_total))
    random.shuffle(indices)

    indices_train = indices[:n_train]
    indices_test = indices[n_train:]

    X_train = X_data[indices_train]
    y_train = y_data[indices_train]
    X_test = X_data[indices_test]
    y_test = y_data[indices_test]

    print(f"Conjunto de entrenamiento: {len(X_train)} muestras")
    print(f"Conjunto de prueba: {len(X_test)} muestras")

    print("Seleccione el método de clasificación:")
    print("1. Mínima Distancia")
    print("2. K-NN")
    metodo = input("Opción (1 o 2): ")

    if metodo == "1":
        centroides = calcular_centroides(X_train, y_train)
        y_pred = [clasificar_minima_distancia(punto, centroides) for punto in X_test]
        print("Usando clasificador de Mínima Distancia")
    else:
        try:
            k = int(input("Ingrese el valor de K: "))
            if k <= 0:
                k = 3
        except:
            k = 3

        y_pred = [clasificar_knn(punto, X_train, y_train, k) for punto in X_test]
        print(f"Usando clasificador K-NN con K={k}")

    accuracy, error, correctos, total = calcular_metricas(y_test, y_pred)

    print(f"Muestras correctas: {correctos}/{total}")
    print(f"Porcentaje de eficiencia: {accuracy:.2f}%")
    print(f"Porcentaje de error: {error:.2f}%")


def k_fold_cross_validation():
    if X_data is None or y_data is None:
        print("Error: No hay datos cargados")
        return

    print("VALIDACIÓN K-FOLD CROSS VALIDATION")

    try:
        k_folds = int(input("Ingrese el número de grupos (K): "))
        if k_folds <= 1:
            k_folds = 5
            print("Valor inválido. Usando K=5 por defecto.")
    except:
        k_folds = 5
        print("Valor inválido. Usando K=5 por defecto.")

    print("Seleccione el método de clasificación:")
    print("1. Mínima Distancia")
    print("2. K-NN")
    metodo = input("Opción (1 o 2): ")

    k_nn = 3
    if metodo == "2":
        try:
            k_nn = int(input("Ingrese el valor de K para K-NN: "))
            if k_nn <= 0:
                k_nn = 3
        except:
            k_nn = 3

    n_total = len(X_data)
    indices = list(range(n_total))
    random.shuffle(indices)

    fold_size = n_total // k_folds
    folds = []

    for i in range(k_folds):
        inicio = i * fold_size
        if i == k_folds - 1:
            fin = n_total
        else:
            fin = (i + 1) * fold_size
        folds.append(indices[inicio:fin])

    accuracies = []
    errors = []

    print(f"Ejecutando {k_folds}-Fold Cross Validation...")

    for i in range(k_folds):
        print(f"Fold {i + 1}")

        test_indices = folds[i]
        train_indices = []
        for j in range(k_folds):
            if j != i:
                train_indices.extend(folds[j])

        X_train = X_data[train_indices]
        y_train = y_data[train_indices]
        X_test = X_data[test_indices]
        y_test = y_data[test_indices]

        print(f"Entrenamiento: {len(X_train)} muestras")
        print(f"Prueba: {len(X_test)} muestras")

        if metodo == "1":
            centroides = calcular_centroides(X_train, y_train)
            y_pred = [clasificar_minima_distancia(punto, centroides) for punto in X_test]
        else:
            y_pred = [clasificar_knn(punto, X_train, y_train, k_nn) for punto in X_test]

        accuracy, error, correctos, total = calcular_metricas(y_test, y_pred)
        accuracies.append(accuracy)
        errors.append(error)

        print(f"Correctas: {correctos}/{total}")
        print(f"Eficiencia: {accuracy:.2f}%")
        print(f"Error: {error:.2f}%")

    accuracy_promedio = sum(accuracies) / len(accuracies)
    error_promedio = sum(errors) / len(errors)

    accuracy_varianza = sum((acc - accuracy_promedio) ** 2 for acc in accuracies) / len(accuracies)
    accuracy_std = math.sqrt(accuracy_varianza)

    error_varianza = sum((err - error_promedio) ** 2 for err in errors) / len(errors)
    error_std = math.sqrt(error_varianza)

    print("RESULTADOS GENERALES")
    print(f"Eficiencia promedio: {accuracy_promedio:.2f}% ± {accuracy_std:.2f}%")
    print(f"Error promedio: {error_promedio:.2f}% ± {error_std:.2f}%")

    print(f"Detalle por fold:")
    for i, (acc, err) in enumerate(zip(accuracies, errors)):
        print(f"Fold {i + 1}: Eficiencia = {acc:.2f}%, Error = {err:.2f}%")


def bootstrap_validation():
    if X_data is None or y_data is None:
        print("Error: No hay datos cargados")
        return

    print("VALIDACIÓN BOOTSTRAP")

    try:
        k_experimentos = int(input("Ingrese la cantidad de experimentos (K): "))
        if k_experimentos <= 0:
            k_experimentos = 10
    except:
        k_experimentos = 10
        print("Valor inválido. Usando 10 experimentos por defecto.")

    try:
        n_train = int(input("Ingrese la cantidad de muestras en el conjunto de aprendizaje: "))
        if n_train <= 0 or n_train > len(X_data):
            n_train = len(X_data) // 2
    except:
        n_train = len(X_data) // 2
        print(f"Valor inválido. Usando {n_train} muestras por defecto.")

    try:
        n_test = int(input("Ingrese la cantidad de muestras en el conjunto de prueba: "))
        if n_test <= 0 or n_test > len(X_data):
            n_test = len(X_data) // 4
    except:
        n_test = len(X_data) // 4
        print(f"Valor inválido. Usando {n_test} muestras por defecto.")

    print("Seleccione el método de clasificación:")
    print("1. Mínima Distancia")
    print("2. K-NN")
    metodo = input("Opción (1 o 2): ")

    k_nn = 3
    if metodo == "2":
        try:
            k_nn = int(input("Ingrese el valor de K para K-NN: "))
            if k_nn <= 0:
                k_nn = 3
        except:
            k_nn = 3

    accuracies = []
    errors = []
    clases_unicas = np.unique(y_data)

    accuracy_por_clase = {clase: [] for clase in clases_unicas}
    error_por_clase = {clase: [] for clase in clases_unicas}

    print(f"Ejecutando {k_experimentos} experimentos Bootstrap...")

    for i in range(k_experimentos):
        print(f"Experimento {i + 1}")

        indices_train = [random.randint(0, len(X_data) - 1) for _ in range(n_train)]
        X_train = X_data[indices_train]
        y_train = y_data[indices_train]

        indices_disponibles = list(range(len(X_data)))
        indices_test = random.sample(indices_disponibles, min(n_test, len(indices_disponibles)))
        X_test = X_data[indices_test]
        y_test = y_data[indices_test]

        print(f"Entrenamiento: {len(X_train)} muestras (con reemplazo)")
        print(f"Prueba: {len(X_test)} muestras (sin reemplazo)")

        if metodo == "1":
            centroides = calcular_centroides(X_train, y_train)
            y_pred = [clasificar_minima_distancia(punto, centroides) for punto in X_test]
        else:
            y_pred = [clasificar_knn(punto, X_train, y_train, k_nn) for punto in X_test]

        accuracy, error, correctos, total = calcular_metricas(y_test, y_pred)
        accuracies.append(accuracy)
        errors.append(error)

        print(f"Correctas: {correctos}/{total}")
        print(f"Eficiencia: {accuracy:.2f}%")
        print(f"Error: {error:.2f}%")

        for clase in clases_unicas:
            indices_clase = [j for j, c in enumerate(y_test) if c == clase]
            if indices_clase:
                y_true_clase = [y_test[j] for j in indices_clase]
                y_pred_clase = [y_pred[j] for j in indices_clase]

                acc_clase, err_clase, _, _ = calcular_metricas(y_true_clase, y_pred_clase)
                accuracy_por_clase[clase].append(acc_clase)
                error_por_clase[clase].append(err_clase)

                print(f"  Clase {clase}: {acc_clase:.2f}% eficiencia")

    accuracy_promedio = sum(accuracies) / len(accuracies)
    error_promedio = sum(errors) / len(errors)

    accuracy_varianza = sum((acc - accuracy_promedio) ** 2 for acc in accuracies) / len(accuracies)
    accuracy_std = math.sqrt(accuracy_varianza)

    error_varianza = sum((err - error_promedio) ** 2 for err in errors) / len(errors)
    error_std = math.sqrt(error_varianza)

    print("RESULTADOS GENERALES")
    print(f"Eficiencia promedio: {accuracy_promedio:.2f}% ± {accuracy_std:.2f}%")
    print(f"Error promedio: {error_promedio:.2f}% ± {error_std:.2f}%")

    print(f"Resultados por clase:")
    for clase in clases_unicas:
        if accuracy_por_clase[clase]:
            acc_clase_prom = sum(accuracy_por_clase[clase]) / len(accuracy_por_clase[clase])
            err_clase_prom = sum(error_por_clase[clase]) / len(error_por_clase[clase])

            acc_clase_var = sum((acc - acc_clase_prom) ** 2 for acc in accuracy_por_clase[clase]) / len(
                accuracy_por_clase[clase])
            acc_clase_std = math.sqrt(acc_clase_var)

            err_clase_var = sum((err - err_clase_prom) ** 2 for err in error_por_clase[clase]) / len(
                error_por_clase[clase])
            err_clase_std = math.sqrt(err_clase_var)

            print(f"Clase {clase}:")
            print(f"  Eficiencia: {acc_clase_prom:.2f}% ± {acc_clase_std:.2f}%")
            print(f"  Error: {err_clase_prom:.2f}% ± {err_clase_std:.2f}%")


def main():
    print("MÉTODOS DE VALIDACIÓN")
    print("Train/Test, K-Fold Cross Validation, Bootstrap")

    while True:
        print("MENÚ PRINCIPAL")
        print("1. Cargar archivo de datos")
        print("2. Validación Train and Test")
        print("3. Validación K-Fold Cross Validation")
        print("4. Validación Bootstrap")
        print("5. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == '1':
            nombre_archivo = input("Ingrese el nombre del archivo CSV: ")
            datos = cargar_archivo(nombre_archivo)
            if datos is not None:
                seleccionar_columnas(datos)

        elif opcion == '2':
            train_test_validation()

        elif opcion == '3':
            k_fold_cross_validation()

        elif opcion == '4':
            bootstrap_validation()

        elif opcion == '5':
            print("¡Hasta luego!")
            break

        else:
            print("Opción no válida. Intente de nuevo.")


if __name__ == "__main__":
    main()