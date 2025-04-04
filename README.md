# MapLine 1.0.9

## Description

**MapLine** is an emission line fitting software optimized for active galactic nuclei (AGN) spectra, whether one-dimensional or obtained through integral field spectroscopy (IFS). Its main purpose is to characterize the broad and narrow components of various emission lines in the optical range. It allows modifying aspects such as the number of components to fit, the lines to consider, the flux profile (Gaussian, double Gaussian, Lorentzian, skew), the spectral fitting range, the input/output file type, continuum extraction, among others. The software is written in Python and has a modular structure, making it easy to customize and adapt to different analysis needs.

## Installation

First, verify the Python version installed on your system. MapLine requires Python 3.12.3 or higher.

It is recommended to create a virtual environment for installing MapLine and the necessary dependencies. This allows better control of the environment and avoids conflicts with other installed packages. You can create a virtual environment with the following commands:

```bash
# Create a folder for virtual environments
mkdir .venvs

# Create a virtual environment named 'MyEnv'
python -m venv .venvs/MyEnv

# Activate the virtual environment
source .venvs/MyEnv/bin/activate
```

Once the virtual environment is activated, you can install MapLine with the following command:

```bash
pip install mapline
```

During the installation, all necessary dependencies will be installed.

## Project Structure

MapLine is composed of different modules that contain the functions needed for emission line fitting:

- **line_fit.py**: Main module that contains the fitting functions for one-dimensional spectra (`line_fit_single`) and data cubes (`line_fit`).
- **mcmc.py**: Implements the Monte Carlo – Markov Chain (MCMC) algorithm to optimize emission line fitting using the `emcee` package.
- **models.py**: Contains the emission line models used in the fitting, such as `emission_line_model` and `line_model`.
- **priors.py**: Provides statistical functions to estimate the likelihood and initial parameter values for fitting.
- **tools.py**: Includes additional tools such as reading `.fits` files and handling configuration files.

## Usage

MapLine is run from the command line using the `run_mapline` command. The basic execution structure is as follows:

```bash
run_mapline [options]
```

Some available commands are:

- `run`: Runs MapLine to fit emission lines.
- `runoned`: Gets the spectrum model.

### Options

| Option            | Description                                  |
|-------------------|----------------------------------------------|
| `-g, --config_file` | Configuration file name.                    |
| `-n, --name`      | Data cube name.                              |
| `-o, --name_out`  | Output file name.                            |
| `-m, --mask`      | Mask file name.                              |
| `-p, --path`      | Path to the data cube.                       |
| `-y, --path_out`  | Path to the output files.                    |
| `-c, --ncpus`     | Number of CPUs to use.                       |
| `-d, --double`    | Enable double Gaussian mode.                 |
| `-k, --kskew`     | Enable skew line mode.                       |
| `-t, --test`      | Test mode.                                   |
| `-e, --error`     | Calculate error vector.                      |
| `-z, --zt`        | Object redshift.                             |

For example, to run MapLine using a configuration file named `config.yml`, using 6 CPUs, the double Gaussian model, and a line configuration file named `line_prop.yml`, the command would be:

```bash
run_mapline -g config.yml -c 6 -d -q line_prop.yml
```

All these options, along with additional ones, can also be specified in the configuration file.

## Configuration Files

MapLine allows defining several parameters through configuration files, such as `config.yml` and `line_prop.yml`. These files allow full customization of the settings and options of the program to meet the user's specific needs.

## Contributions

Contributions to MapLine are welcome. You can submit your suggestions, bug reports, or improvements through the official GitHub repository.

## License

MapLine is distributed under the MIT license. You are free to use, modify, and distribute it, as long as proper attribution is maintained.

## Español
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
