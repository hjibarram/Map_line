# MapLine 1.0.7

## Descripción

**MapLine** es un software de ajuste de líneas de emisión optimizado para espectros de núcleos activos de galaxias (AGN), ya sean unidimensionales o obtenidos por espectroscopía de campo integral (IFS). Su principal propósito es caracterizar los componentes anchos y angostos de varias líneas de emisión en el rango óptico. Permite modificar aspectos como el número de componentes a ajustar, las líneas a considerar, el perfil de flujo (gaussiano, doble gaussiano, lorentziano, skew), el rango espectral de ajuste, el tipo de archivo de entrada/salida, la extracción del continuo, entre otros. El software está escrito en Python y tiene una estructura modular, lo cual facilita su personalización y adaptación a diferentes necesidades de análisis.

## Instalación

Primero, verifica la versión de Python instalada en tu equipo. MapLine requiere Python 3.12.3 o superior.

Es recomendable crear un entorno virtual para la instalación de MapLine y las dependencias necesarias. Esto permite un mejor control del entorno y evita conflictos con otros paquetes instalados. Puedes crear un entorno virtual con los siguientes comandos:

```bash
# Crear una carpeta para entornos virtuales
mkdir .venvs

# Crear un entorno virtual llamado 'MyEnv'
python -m venv .venvs/MyEnv

# Activar el entorno virtual
source .venvs/MyEnv/bin/activate
```

Una vez activado el entorno virtual, puedes instalar MapLine con el siguiente comando:

```bash
pip install mapline
```

Durante la instalación se instalarán todas las dependencias necesarias.

## Estructura del Proyecto

MapLine está compuesto por distintos módulos que contienen las funciones necesarias para los ajustes de líneas de emisión:

- **line_fit.py**: Módulo principal que contiene las funciones de ajuste para espectros en una dimensión (`line_fit_single`) y para cubos de datos (`line_fit`).
- **mcmc.py**: Implementa el algoritmo Monte Carlo – Markov Chain (MCMC) para optimizar los ajustes de líneas de emisión usando la paquetería `emcee`.
- **models.py**: Contiene los modelos de líneas de emisión utilizados en los ajustes, como `emission_line_model` y `line_model`.
- **priors.py**: Proporciona funciones estadísticas para estimar la confiabilidad (likelihood) y los valores iniciales de los parámetros de ajuste.
- **tools.py**: Incluye herramientas adicionales como la lectura de archivos en formato `.fits` y manejo de archivos de configuración.

## Uso

MapLine se ejecuta desde la línea de comandos utilizando la instrucción `run_mapline`. La estructura básica de ejecución es la siguiente:

```bash
run_mapline [opciones]
```

Algunos de los comandos disponibles son:

- `run`: Ejecuta MapLine para ajustar líneas de emisión.
- `runoned`: Obtiene el modelo del espectro.

### Opciones

| Opción           | Descripción                                  |
|------------------|----------------------------------------------|
| `-g, --config_file` | Nombre del archivo de configuración.       |
| `-n, --name`     | Nombre del cubo de datos.                    |
| `-o, --name_out` | Nombre de los archivos de salida.            |
| `-m, --mask`     | Nombre del archivo de 'máscara'.             |
| `-p, --path`     | Ruta al cubo de datos.                       |
| `-y, --path_out` | Ruta de los archivos de salida.              |
| `-c, --ncpus`    | Número de CPUs a utilizar.                   |
| `-d, --double`   | Activar el modo de doble gaussiana.          |
| `-k, --kskew`    | Activar el modo skew line.                   |
| `-t, --test`     | Modo de prueba.                              |
| `-e, --error`    | Calcular el vector de errores.               |
| `-z, --zt`       | Redshift del objeto.                         |

Por ejemplo, para ejecutar MapLine usando un archivo de configuración llamado `config.yml`, utilizando 6 CPUs, el modelo de doble gaussiana, y un archivo de configuración de líneas de emisión llamado `line_prop.yml`, el comando sería:

```bash
run_mapline -g config.yml -c 6 -d -q line_prop.yml
```

Todas estas opciones, junto con otras adicionales, también pueden ser especificadas en el archivo de configuración.

## Archivos de Configuración

MapLine permite definir varios parámetros a través de archivos de configuración, como `config.yml` y `line_prop.yml`. Estos archivos permiten personalizar completamente los ajustes y opciones del programa para adaptarse a las necesidades específicas del usuario.

## Contribuciones

Las contribuciones a MapLine son bienvenidas. Puedes enviar tus sugerencias, errores encontrados, o mejoras a través del repositorio oficial en GitHub.

## Licencia

MapLine está distribuido bajo la licencia MIT. Puedes usarlo, modificarlo y distribuirlo libremente siempre y cuando se mantenga la atribución correspondiente.
