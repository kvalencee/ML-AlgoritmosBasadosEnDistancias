import traceback


def cargarDatos(ruta, delimitador):
    matriz = []
    try:
        with open(ruta, 'r') as base:
            for linea in base:
                lineaLimpia = linea.strip()
                if lineaLimpia != "":
                    atributos = lineaLimpia.split(delimitador)
                    matriz.append(atributos)
        return matriz
    except FileNotFoundError:
        return 0


def determinarTipoDato(valor):
    valor = valor.strip()

    if valor == "":
        return "vacio"

    try:
        int(valor)
        return "entero"
    except ValueError:
        pass

    try:
        float(valor)
        return "flotante"
    except ValueError:
        pass

    if valor.lower() in ['true', 'false', 'verdadero', 'falso', '1', '0']:
        return "booleano"

    return "cadena"


def definirTipoYMedida(matriz):
    if not matriz:
        return [], []

    num_columnas = len(matriz[0])
    tipos_columnas = []
    medidas_columnas = []

    for col in range(num_columnas):
        tipos_en_columna = {}
        valores_columna = []

        for fila in matriz:
            if col < len(fila):
                valor = fila[col]
                tipo = determinarTipoDato(valor)

                if tipo in tipos_en_columna:
                    tipos_en_columna[tipo] += 1
                else:
                    tipos_en_columna[tipo] = 1

                valores_columna.append(valor.strip())

        tipo_predominante = max(tipos_en_columna, key=tipos_en_columna.get)
        tipos_columnas.append(tipo_predominante)

        if tipo_predominante in ["entero", "flotante"]:
            valores_numericos = []
            for valor in valores_columna:
                try:
                    if tipo_predominante == "entero":
                        valores_numericos.append(int(valor))
                    else:
                        valores_numericos.append(float(valor))
                except ValueError:
                    continue

            if len(set(valores_numericos)) <= 10:
                medidas_columnas.append("discreta")
            else:
                medidas_columnas.append("continua")
        else:
            medidas_columnas.append("nominal")

    return tipos_columnas, medidas_columnas


def seleccionarAtributos(matriz, indices_columnas):
    if not matriz or not indices_columnas:
        return []

    matriz_reducida = []
    for fila in matriz:
        nueva_fila = []
        for indice in indices_columnas:
            if 0 <= indice < len(fila):
                nueva_fila.append(fila[indice])
        if nueva_fila:
            matriz_reducida.append(nueva_fila)

    return matriz_reducida


def seleccionarRenglones(matriz, indices_filas):
    if not matriz or not indices_filas:
        return []

    matriz_reducida = []
    for indice in indices_filas:
        if 0 <= indice < len(matriz):
            matriz_reducida.append(matriz[indice])

    return matriz_reducida


def guardarMatriz(matriz, nombre_archivo):
    try:
        with open(nombre_archivo, 'w') as archivo:
            for fila in matriz:
                linea = ','.join(str(elemento) for elemento in fila)
                archivo.write(linea + '\n')
        print(f"Matriz guardada exitosamente en: {nombre_archivo}")
        return True
    except Exception as e:
        print(f"Error al guardar archivo: {e}")
        return False


def imprimirColumnaClase(matriz, columna_clase):
    print("Columna CLASE completa:")
    for i in range(len(matriz)):
        if columna_clase < len(matriz[i]):
            print(matriz[i][columna_clase])


def clasificarVinoPorCalidad(matriz, columna_clase):
    print("Clasificación del vino por calidad:")
    for i in range(len(matriz)):
        if columna_clase < len(matriz[i]):
            try:
                calidad = int(float(matriz[i][columna_clase]))
                if calidad <= 5:
                    print(f"Vino de baja calidad ({calidad})")
                elif calidad == 6:
                    print(f"Vino de calidad media ({calidad})")
                else:
                    print(f"Vino de alta calidad ({calidad})")
            except ValueError:
                print("No se pudo convertir la calidad:", matriz[i][columna_clase])


def mostrarEstadisticas(matriz, tipos, medidas):
    if not matriz:
        print("No hay datos para mostrar estadísticas")
        return

    print("ESTADÍSTICAS DEL ARCHIVO:")
    print(f"Número de patrones (filas): {len(matriz)}")
    print(f"Número de atributos (columnas): {len(matriz[0]) if matriz else 0}")

    print("TIPOS DE DATOS POR COLUMNA:")
    for i, (tipo, medida) in enumerate(zip(tipos, medidas)):
        print(f"Columna {i + 1}: Tipo = {tipo}, Medida = {medida}")


def parsearIndices(entrada):
    try:
        indices = []
        partes = entrada.split(',')
        for parte in partes:
            parte = parte.strip()
            if '-' in parte:
                inicio, fin = parte.split('-')
                inicio = int(inicio) - 1
                fin = int(fin) - 1
                for i in range(inicio, fin + 1):
                    indices.append(i)
            else:
                indices.append(int(parte) - 1)
        return indices
    except:
        return []


def main():
    print("SISTEMA DE CARGA DE DATOS")

    print("Ingresa el nombre del archivo")
    ruta = input()
    print("Ingresa el delimitador del archivo")
    delimitador = input()

    matriz = cargarDatos(ruta, delimitador)
    if matriz:
        patrones = len(matriz)
        atributos = len(matriz[0]) if matriz else 0

        print(f"El numero de patrones es: {patrones}")
        print(f"El numero de atributos es: {atributos}")

        tipos, medidas = definirTipoYMedida(matriz)
        mostrarEstadisticas(matriz, tipos, medidas)

        while True:
            print("\nMENU PRINCIPAL:")
            print("1. Mostrar contenido de la matriz")
            print("2. Seleccionar y mostrar columna clase")
            print("3. Clasificar vino por calidad (solo para archivos de vino)")
            print("4. Seleccionar subconjunto de atributos")
            print("5. Seleccionar subconjunto de renglones")
            print("6. Guardar matriz reducida")
            print("7. Salir")

            opcion = input("Seleccione una opción: ")

            if opcion == "1":
                print("Contenido de la matriz:")
                for i, fila in enumerate(matriz):
                    print(f"Fila {i + 1}: {fila}")

            elif opcion == "2":
                print("Especifica la columna CLASE (1 a {}):".format(atributos))
                try:
                    clase = int(input()) - 1
                    if 0 <= clase < atributos:
                        imprimirColumnaClase(matriz, clase)
                    else:
                        print("Número de columna inválido")
                except ValueError:
                    print("Entrada inválida")

            elif opcion == "3":
                if "wine" in ruta.lower():
                    print("Especifica la columna de calidad (1 a {}):".format(atributos))
                    try:
                        clase = int(input()) - 1
                        if 0 <= clase < atributos:
                            clasificarVinoPorCalidad(matriz, clase)
                        else:
                            print("Número de columna inválido")
                    except ValueError:
                        print("Entrada inválida")
                else:
                    print("Esta función solo está disponible para archivos de vino")

            elif opcion == "4":
                print(f"Ingrese los números de columnas a seleccionar (1 a {atributos}):")
                print("Formato: números separados por comas, o rangos con guión (ej: 1,3,5-8)")
                entrada = input()
                indices = parsearIndices(entrada)

                if indices:
                    indices_validos = [i for i in indices if 0 <= i < atributos]
                    if indices_validos:
                        matriz_reducida = seleccionarAtributos(matriz, indices_validos)
                        print(f"Matriz reducida creada con {len(indices_validos)} columnas")
                        print("Primeras 5 filas de la matriz reducida:")
                        for i, fila in enumerate(matriz_reducida[:5]):
                            print(f"Fila {i + 1}: {fila}")

                        guardar = input("¿Desea guardar esta matriz? (s/n): ")
                        if guardar.lower() == 's':
                            nombre = input("Ingrese el nombre del archivo (sin extensión): ")
                            guardarMatriz(matriz_reducida, f"{nombre}_atributos.txt")
                    else:
                        print("No se encontraron índices válidos")
                else:
                    print("Formato de entrada inválido")

            elif opcion == "5":
                print(f"Ingrese los números de filas a seleccionar (1 a {patrones}):")
                print("Formato: números separados por comas, o rangos con guión (ej: 1,3,5-10)")
                entrada = input()
                indices = parsearIndices(entrada)

                if indices:
                    indices_validos = [i for i in indices if 0 <= i < patrones]
                    if indices_validos:
                        matriz_reducida = seleccionarRenglones(matriz, indices_validos)
                        print(f"Matriz reducida creada con {len(indices_validos)} filas")
                        print("Primeras 5 filas de la matriz reducida:")
                        for i, fila in enumerate(matriz_reducida[:5]):
                            print(f"Fila {i + 1}: {fila}")

                        guardar = input("¿Desea guardar esta matriz? (s/n): ")
                        if guardar.lower() == 's':
                            nombre = input("Ingrese el nombre del archivo (sin extensión): ")
                            guardarMatriz(matriz_reducida, f"{nombre}_filas.txt")
                    else:
                        print("No se encontraron índices válidos")
                else:
                    print("Formato de entrada inválido")

            elif opcion == "6":
                nombre = input("Ingrese el nombre del archivo (sin extensión): ")
                guardarMatriz(matriz, f"{nombre}_completo.txt")

            elif opcion == "7":
                print("¡Hasta luego!")
                break

            else:
                print("Opción no válida")

    else:
        print("El archivo no se encuentra")


if __name__ == "__main__":
    main()