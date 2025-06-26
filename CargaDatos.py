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


# Clasificación del vino tinto
def imprimirColumnaClase(matriz, columna_clase):
    """
    Imprime todos los valores de la columna que el usuario define como clase (etiqueta).
    """
    print("\nColumna CLASE completa:")
    for i in range(len(matriz)):
        print(matriz[i][columna_clase])


def clasificarVinoPorCalidad(matriz, columna_clase):
    """
    Clasifica vinos según su calidad (int) como baja, media o alta.
    Solo debe usarse si el archivo es del vino.
    """
    print("\nClasificación del vino por calidad:")
    for i in range(len(matriz)):
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


def main():
    print("Ingresa el nombre del archivo")
    ruta = input()
    print("Ingresa el delimitador del archivo")
    delimitador = input()

    matriz = cargarDatos(ruta, delimitador)
    if matriz:
        patrones = len(matriz)
        atributos = len(matriz[0])

        print(f"El numero de patrones es: {patrones}")
        print(f"El numero de atributos es: {atributos}")
        # print("El contenido de la matriz es:")
        # for i in range(patrones):
        #     print(matriz[i])

        print("Especifica la columna CLASE: ")
        clase = int(input())

        imprimirColumnaClase(matriz, clase)
        if "wine" in ruta.lower():
            clasificarVinoPorCalidad(matriz, clase)

        # print(type(matriz))

        print(matriz[1][clase])

    else:
        print(f"El archivo no se encuentra")

    # print("Columna clase de todos los atributos: ")
    # for i in range(atributos):
    #     print(f"{i}. {matriz[i[clase]]} ")


main()
