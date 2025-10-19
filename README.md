# Proyecto de Clasificador de Sueño

Este proyecto se desarrolló en Google Colab y se dividió en dos etapas principales:

## Primera Parte

En esta etapa se realizó una primera aproximación al problema de clasificación de sueño utilizando únicamente 7 features del primer canal EEG:

- Potencia de bandas

- Promedio de la señal

- Desviación estándar de la señal

Con estas features se entrenaron modelos de MLP y CNN + LSTM.

## Segunda Parte

Para mejorar la precisión del modelo, se extrajeron más features de los diferentes canales disponibles en los datos. En total se obtuvieron 31 features.
Posteriormente, se busca reentrenar los modelos de la primera parte ajustando los hiperparámetros y evaluando su desempeño. (Trabajo en curso)

### Features

- 7 Features iniciales: Descargar dataframe de 7 features

- 31 Features ampliadas: Descargar dataframe de 31 features
