# Documentación

API con modelo de predicción de números usando imagenes, el modelo está hecho con una red neuronal con capas convolucionales usando la libreria `tensorflow`.

El modelo no acierta muy bien, tiene problemas en el tratamiento de imagenes escritas a mano, esta api es una práctica de la especialización en Inteligencia Artificial y Big Data.

## Rutas
Existen tres rutas en esta API:
- `HealthCheck`: `GET` Devuelve desde cuando está vivo el servidor (__`/health`__)
- `Image`: `POST` Al enviar una `IMAGEN` devuelve un `json` con la predicción del número enviado. (__`/image`__)
- `Routes`: `GET` Devuelve todas las rutas de la API (__`/`__)