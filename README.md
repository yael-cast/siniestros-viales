# Aplicación de análisis de siniestros viales

## Archivos de datos
Copiar en esta misma carpeta:
- `siniestros_viales_hechos.xlsx`
- `siniestros_viales_victimas.csv`

El CSV usa separador `;`, igual que en Colab.

## Ejecutar en Windows
1. Abrir esta carpeta en Visual Studio Code.
2. Abrir Terminal > Nueva terminal.
3. Instalar dependencias: `python -m pip install -r requirements.txt`
4. Ejecutar: `python -m streamlit run app.py`
5. Abrir en el navegador la dirección local que indique la terminal.

## Incluye
- Filtros por año, comuna y gravedad.
- Estadísticas y gráficos.
- Cálculo de las tres hipótesis.
- Ranking de zonas por comuna y filtro por modo de desplazamiento.

## Notas
Los archivos de datos no están incluidos. Las alertas son informativas y se basan en frecuencias históricas; no predicen accidentes ni garantizan rutas seguras. Comparar los resultados con Colab antes de entregar.
