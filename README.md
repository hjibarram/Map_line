# MapLine 2.0.1

## Description

**MapLine** is an emission line fitting software optimized for active galactic nuclei (AGN) spectra, whether one-dimensional or obtained through integral field spectroscopy (IFS). Its main purpose is to characterize the broad and narrow components of various emission lines in the optical range. It allows modifying aspects such as the number of components to fit, the lines to consider, the flux profile (Gaussian, double Gaussian, Lorentzian, skew), the spectral fitting range, the input/output file type, continuum extraction, among others. The software is written in Python and has a modular structure, making it easy to customize and adapt to different analysis needs.

The package supports both:

- **single-spectrum fitting**
- **spatially resolved IFU cube fitting**

MapLines includes tools for spectral modeling, Bayesian parameter estimation
via MCMC, diagnostic plotting, and spatial analysis of emission-line
properties.

The full documentation can be looked at https://hjibarram.github.io/Map_line/index.html

---

# Documentation

Complete documentation is available through the Sphinx documentation included
in the repository.

To build the documentation locally:

```bash
cd docs
make html
```

The generated documentation will appear in:

```
docs/build/html/index.html
```

The documentation includes:

- Installation guide
- Quickstart tutorial
- Configuration file documentation
- Methodology overview
- Example workflows
- CLI documentation
- Full API reference

---

# Features

MapLines provides several features for emission-line analysis:

- Flexible emission-line modeling
- Gaussian, skewed Gaussian, Lorentzian, and Voigt profiles
- Outflow component modeling
- FeII template fitting
- Bayesian parameter estimation using **MCMC (emcee)**
- IFU cube fitting
- Diagnostic diagram generation (BPT, WHAN, WHAD)
- Spatial extraction tools
- Automated plotting utilities

---

# Installation

MapLines requires **Python ≥ 3.10** and the scientific Python ecosystem.

The recommended installation method is through a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package from the repository:

```bash
pip install -e .
```

Typical dependencies include:

- numpy
- scipy
- matplotlib
- astropy
- emcee
- pyyaml
- corner

---

# Command Line Interface

MapLines provides a command-line interface using **Click**.

The main command is:

```bash
run_mapline
```

Two main workflows are available.

---

## IFU cube fitting

```bash
run_mapline run -g config.yml
```

This command fits emission lines across an IFU data cube.

Outputs typically include:

- model FITS cubes
- parameter FITS cubes
- diagnostic plots

---

## Single-spectrum fitting

```bash
run_mapline runoned -g config.yml
```

This runs the fitting procedure on a single spectrum.

---

# Configuration Files

MapLines uses YAML configuration files to define the spectral model and
fitting setup.

Typical configuration files include:

- `config.yml`
- `line_prop.yml`

These files define:

- emission lines to fit
- model components
- parameter priors
- fitting wavelength ranges

---

# Project Structure

The core functionality is organized in the following modules:

```
MapLines
│
├── line_fit.py
│   Main spectral fitting routines
│
├── models.py
│   Emission-line model definitions
│
├── mcmc.py
│   Bayesian sampling utilities
│
├── priors.py
│   Likelihood and prior functions
│
├── tools.py
│   General utilities and data extraction tools
│
└── plot_tools.py
    Visualization and diagnostic plotting
```

---

# Example Workflows

Example configurations and scripts are provided in:

```
examples/
├── single_spectra
├── ifu_spectra
└── notebooks
```

These demonstrate typical use cases such as:

- emission-line fitting
- IFU cube analysis
- diagnostic diagram generation
- spatial extraction of spectra

---

# Documentation Structure

The Sphinx documentation is organized into two main sections.

## User Guide

- Installation
- Quickstart
- Configuration
- Examples
- CLI usage
- Methodology

## API Reference

- MapLines modules
- tools
- plot_tools
- mcmc
- priors

---

# Contributing

Contributions are welcome.

You can contribute by:

- reporting bugs
- suggesting improvements
- adding features
- improving documentation

Please open an issue or pull request in the GitHub repository.

---

# License

MapLines is distributed under the **MIT License**.

You are free to use, modify, and distribute it provided that proper
attribution is maintained.

---




## Contributions

Contributions to MapLine are welcome. You can submit your suggestions, bug reports, or improvements through the official GitHub repository.

## License

MapLine is distributed under the MIT license. You are free to use, modify, and distribute it, as long as proper attribution is maintained.

# Acknowledgements

MapLines was developed for the analysis of emission-line spectra in
galaxies and AGN, with particular emphasis on applications to
integral-field spectroscopy surveys. HIM acknowledge the support from grants SECIHTI CBF2023-2024-1418 and CF-2023-G-543, PAPIIT UNAM IA104325, IN-106823, and IN-119123

## Español
## Descripción

**MapLine** es un software de ajuste de líneas de emisión optimizado para espectros de núcleos activos de galaxias (AGN), ya sean unidimensionales o obtenidos por espectroscopía de campo integral (IFS). Su principal propósito es caracterizar los componentes anchos y angostos de varias líneas de emisión en el rango óptico. Permite modificar aspectos como el número de componentes a ajustar, las líneas a considerar, el perfil de flujo (gaussiano, doble gaussiano, lorentziano, skew), el rango espectral de ajuste, el tipo de archivo de entrada/salida, la extracción del continuo, entre otros. El software está escrito en Python y tiene una estructura modular, lo cual facilita su personalización y adaptación a diferentes necesidades de análisis.

El paquete permite trabajar tanto con:

- **ajuste de espectros individuales**
- **ajuste de cubos IFU con resolución espacial**

MapLines incluye herramientas para el modelado espectral, estimación
bayesiana de parámetros mediante MCMC, generación de gráficos de diagnóstico
y análisis espacial de propiedades de líneas de emisión.

---

# Documentación

La documentación completa está disponible a través de la documentación de
Sphinx incluida en el repositorio.

Para generar la documentación localmente:

```bash
cd docs
make html
```

La documentación generada aparecerá en:

```
docs/build/html/index.html
```

La documentación incluye:

- Guía de instalación
- Tutorial de inicio rápido
- Documentación de archivos de configuración
- Descripción de la metodología
- Flujos de trabajo de ejemplo
- Documentación de la interfaz de línea de comandos (CLI)
- Referencia completa de la API

---

# Características

MapLines proporciona varias funcionalidades para el análisis de líneas de
emisión:

- Modelado flexible de líneas de emisión
- Perfiles Gaussianos, Gaussianos asimétricos (skewed), Lorentzianos y Voigt
- Modelado de componentes de outflow
- Ajuste de plantillas de FeII
- Estimación bayesiana de parámetros mediante **MCMC (emcee)**
- Ajuste de cubos IFU
- Generación de diagramas de diagnóstico (BPT, WHAN, WHAD)
- Herramientas de extracción espacial
- Utilidades automáticas de visualización

---

# Instalación

MapLines requiere **Python ≥ 3.10** y el ecosistema científico de Python.

El método de instalación recomendado es mediante un entorno virtual.

```bash
python -m venv .venv
source .venv/bin/activate
```

Instalar el paquete desde el repositorio:

```bash
pip install -e .
```

Las dependencias típicas incluyen:

- numpy
- scipy
- matplotlib
- astropy
- emcee
- pyyaml
- corner

---

# Interfaz de Línea de Comandos

MapLines proporciona una interfaz de línea de comandos utilizando **Click**.

El comando principal es:

```bash
run_mapline
```

Existen dos flujos de trabajo principales.

---

## Ajuste de cubos IFU

```bash
run_mapline run -g config.yml
```

Este comando ajusta líneas de emisión a lo largo de un cubo de datos IFU.

Las salidas típicamente incluyen:

- cubos FITS del modelo
- cubos FITS de parámetros
- gráficos de diagnóstico

---

## Ajuste de espectro individual

```bash
run_mapline runoned -g config.yml
```

Este comando ejecuta el procedimiento de ajuste sobre un espectro individual.

---

# Archivos de Configuración

MapLines utiliza archivos de configuración en formato YAML para definir el
modelo espectral y la configuración del ajuste.

Los archivos típicos incluyen:

- `config.yml`
- `line_prop.yml`

Estos archivos definen:

- líneas de emisión a ajustar
- componentes del modelo
- priors de los parámetros
- rangos de longitud de onda para el ajuste

---

# Estructura del Proyecto

La funcionalidad principal está organizada en los siguientes módulos:

```
MapLines
│
├── line_fit.py
│   Rutinas principales de ajuste espectral
│
├── models.py
│   Definición de modelos de líneas de emisión
│
├── mcmc.py
│   Utilidades para muestreo bayesiano
│
├── priors.py
│   Funciones de verosimilitud y priors
│
├── tools.py
│   Utilidades generales y herramientas de extracción de datos
│
└── plot_tools.py
    Visualización y gráficos de diagnóstico
```

---

# Flujos de Trabajo de Ejemplo

Ejemplos de configuraciones y scripts se encuentran en:

```
examples/
├── single_spectra
├── ifu_spectra
└── notebooks
```

Estos demuestran casos de uso típicos como:

- ajuste de líneas de emisión
- análisis de cubos IFU
- generación de diagramas de diagnóstico
- extracción espacial de espectros

---

# Estructura de la Documentación

La documentación generada con Sphinx está organizada en dos secciones
principales.

## Guía de Usuario

- Instalación
- Inicio rápido
- Configuración
- Ejemplos
- Uso del CLI
- Metodología

## Referencia de la API

- Módulos de MapLines
- tools
- plot_tools
- mcmc
- priors

---

# Contribuciones

Las contribuciones son bienvenidas.

Puedes contribuir mediante:

- reporte de errores
- sugerencias de mejora
- desarrollo de nuevas funcionalidades
- mejora de la documentación

Por favor abre un issue o un pull request en el repositorio de GitHub.

---

# Licencia

MapLines se distribuye bajo la **Licencia MIT**.

Eres libre de usar, modificar y distribuir el código siempre que se mantenga
la atribución correspondiente.

---

# Agradecimientos

MapLines fue desarrollado para el análisis de espectros de líneas de emisión
en galaxias y AGN, con especial énfasis en aplicaciones a observaciones de
espectroscopía de campo integral. HIM agradece el apoyo de los proyectos SECIHTI CBF2023-2024-1418 y CF-2023-G-543, así como de los proyectos PAPIIT de la UNAM IA104325, IN-106823 e IN-119123.