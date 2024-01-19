# API REST CLASIFICACIÓN DE IMÁGENES CON FLASK

## Objetivo

Desarrollar una API REST en Flask que permita a los usuarios subir imágenes a
través de un endpoint. La API debe utilizar una red neuronal para clasificar las imágenes subidas, controlar el tipo de archivo recibido, y devolver los resultados de la clasificación en formato JSON. Es fundamental implementar los estándares de API REST y utilizar adecuadamente los códigos de estado HTTP para respuestas y errores.

Tendréis que entrenar también el modelo y adjuntarlo en un notebook Jupiter o google colab.

## Requisitos

1) Configuración del Entorno:
    - Instalar Flask y TensorFlow (o cualquier otra librería de aprendizaje automático que prefieran).
    - Asegurarse de que todas las dependencias necesarias estén incluidas en el entorno de trabajo.
    - Utilizar red neuronal identificación dígitos: [Digits Model Link](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)

2) Desarrollo de la API:
    - Crear un endpoint que acepte solo peticiones POST para la carga de imágenes.
    - Crear otro endpoint con un health check tal y como vimos en las lecciones de clase, indicando, el tiempo activo.
    - Utilizar una red neuronal preentrenada (como MobileNet, ResNet, etc.) para clasificar las imágenes, o utilizar las vistas con Ana o Lorenzo si las hay. Pueden usar TensorFlow o Keras.

3) Validación de Archivos:
    - Implementar una verificación para asegurarse de que el archivo subido es una imagen. Considerar tipos de archivo como .jpg, .jpeg, .png, etc.
    - Devolver un error con el código de estado adecuado si el archivo no es una imagen o si falta el archivo.

4) Respuestas de la API:
    - En caso de éxito, devolver un objeto JSON que incluya el nombre de la clase más probable y su confianza.
    - Manejar adecuadamente los errores, como archivos no admitidos, errores del servidor, etc., devolviendo el código de estado HTTP correspondiente y un mensaje de error claro en formato JSON.

5) Documentación:
    - Documentar el endpoint, explicando cómo usarlo, qué tipo de archivos se pueden subir, y el formato de la respuesta JSON.

6) Pruebas:
    - Incluir instrucciones sobre cómo probar el endpoint, por ejemplo, utilizando Postman o un script de Python.

## Criterios de Evaluación
- Funcionalidad: La API debe funcionar según los requisitos y manejar correctamente las imágenes y las predicciones.
- Manejo de Errores: Implementación adecuada de la gestión de errores y devolución de los códigos de estado HTTP correspondientes.
- Calidad del Código: Claridad y organización del código, uso de buenas prácticas de programación.
- Documentación: Claridad en la documentación del endpoint, incluyendo ejemplos de uso.

## Entrega
El proyecto debe entregarse en la plataforma de aules, incluyendo el código fuente, un archivo `README.md` con la documentación y las instrucciones de prueba, y cualquier otro recurso necesario para ejecutar la aplicación (como un archivo requirements.txt para las dependencias de Python), asi como tres imágenes para realizar las pruebas. 

Adjuntar enlace o fichero al google colab o jupiter junto con el entrenamiento.