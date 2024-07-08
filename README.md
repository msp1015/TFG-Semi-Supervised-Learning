# Visualizador de Algoritmos Semi-Supervisados - Versión 2.0

<div align="center">
    <img src="vass2Icon.png" alt="Icono de la Aplicación" width="200"/>
</div>

## Descripción

Esta es una extensión de una aplicación web diseñada para la visualización de algoritmos semisupervisados. La herramienta permite a los usuarios interactuar y comprender mejor cómo funcionan estos algoritmos a través de visualizaciones intuitivas y detalladas.

Para ver el proyecto de la primera versión, consulta el repositorio de [David Martínez Acha](https://github.com/dmacha27/TFG-SemiSupervisado).

## Tecnologías Utilizadas

[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![css](https://img.shields.io/badge/css-1572B6?style=for-the-badge&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![HTML](https://img.shields.io/badge/html-E34F26?style=for-the-badge&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![Javascript](https://img.shields.io/badge/javascript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Bootstrap](https://img.shields.io/badge/bootstrap-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![D3](https://img.shields.io/badge/d3-F9A03C?style=for-the-badge&logo=d3.js&logoColor=white)](https://d3js.org/)
[![Notebook](https://img.shields.io/badge/jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![SonarCloud](https://img.shields.io/badge/SonarCloud-F3702A?style=for-the-badge&logo=sonarcloud&logoColor=white)](https://sonarcloud.io/project/overview?id=msp1015_TFG-Semi-Supervised-Learning)
## Características

- **Interfaz Intuitiva**: Diseñada para ser fácil de usar, permitiendo a los usuarios visualizar y comprender algoritmos semisupervisados sin necesidad de profundos conocimientos técnicos.
- **Visualizaciones Dinámicas**: Utiliza D3.js para generar gráficos interactivos que facilitan la interpretación de los datos y resultados.

## Instalación

Abra una terminal en su equipo y siga los siguienes pasos:

1. Clonar el repositorio
   ```bash
   git clone https://github.com/msp1015/TFG-Semi-Supervised-Learning
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd TFG-Semi-Supervised-Learning
   ```
3. Crear entorno virtual
   ```bash
   python -m venv ./venv
   ```
4. Activa el entorno virtual:

   - En Windows:

     ```bash
     .\venv\Scripts\activate
     ```

   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. Instalar dependencias
   ```bash
   pip install -r requirements.txt
   ```
6. Creación de directorios
   ```bash
   cd web/app
   mkdir runs
   ```
   - En Windows:
   ```bash
    mkdir datasets\anonimos
    mkdir datasets\registrados
   ```
   - En macOS/Linux:
   ```bash
    mkdir datasets/anonimos
    mkdir datasets/registrados
   ```
7. Compilar traducciones (web/app)
   ```bash
   pybabel compile -d translations
   ```

## Uso

1. Ejecuta la aplicación:
   ```bash
   cd ..
   flask run
   ```
2. Abre tu navegador y navega a `http://localhost:5000` para acceder a la aplicación.

3. (Opcional) Añade `--debug` al final del comando `flask run` para entrar en modo desarrollo.

## Contribución

1. Haz un fork del proyecto.
2. Crea una nueva rama con tus cambios:
   ```bash
   git checkout -b feature/nueva-caracteristica
   ```
3. Realiza los cambios y haz commit:
   ```bash
   git commit -m 'Añadir nueva característica'
   ```
4. Sube los cambios a tu fork:
   ```bash
   git push origin feature/nueva-caracteristica
   ```
5. Abre un Pull Request en GitHub.

## Licencia

Este proyecto está bajo la Licencia BSD-3-Clause. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## Autores

- **Mario Sanz Pérez** - _Desarrollador Principal_ - [msp1015](https://github.com/msp1015)
- **Álvar Arnaiz González** - _Tutor / Revisor_ - [alvarag](https://github.com/alvarag)

---

Para cualquier duda o consulta, por favor abre un issue en GitHub o contacta a [msp1015@alu.ubu.es](mailto:msp1015@alu.ubu.es).

### ¡Gracias por usar la aplicación!
