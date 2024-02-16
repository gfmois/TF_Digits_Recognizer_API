# Documentation

API with a number prediction model using images. The model is built with a neural network with convolutional layers using the `tensorflow` library.

The model does not perform very well; it has issues with handwritten images. This API is a practice project for the specialization in Artificial Intelligence and Big Data.

## Routes
There are three routes in this API:
- `HealthCheck`: `GET` Returns the server uptime (__`/health`__)
- `Image`: `POST` When sending an `IMAGE`, it returns a `json` with the prediction of the sent number. (__`/image`__)
- `Routes`: `GET` Returns all the routes of the API (__`/`__)

## Docker
To use it, first download the container with:
> docker pull gfmois/py_digit_class_api
Then lunch your first container with:
> docker run -p 8080:5000 -w /app py_digit_class_api make

# Documentación

API con modelo de predicción de números usando imagenes, el modelo está hecho con una red neuronal con capas convolucionales usando la libreria `tensorflow`.

El modelo no acierta muy bien, tiene problemas en el tratamiento de imagenes escritas a mano, esta api es una práctica de la especialización en Inteligencia Artificial y Big Data.

## Rutas
Existen tres rutas en esta API:
- `HealthCheck`: `GET` Devuelve desde cuando está vivo el servidor (__`/health`__)
- `Image`: `POST` Al enviar una `IMAGEN` devuelve un `json` con la predicción del número enviado. (__`/image`__)
- `Routes`: `GET` Devuelve todas las rutas de la API (__`/`__)

## Docker
Para usarlo, primero descarga el contenedor con el commando:
> docker pull gfmois/py_digit_class_api
Una vez descargado, lanza tu primer contenedor con:
> docker run -p 8080:5000 -w /app py_digit_class_api make
