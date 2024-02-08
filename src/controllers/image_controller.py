from io import BytesIO
from typing import Tuple
from flask import jsonify
from PIL import Image
from werkzeug.datastructures import ImmutableMultiDict, FileStorage
from ..service.ImageService import ImageService

class ImageController:
    def __check_valid_image(self, files: ImmutableMultiDict[str, FileStorage]) -> Tuple[any, int]:
        """
        Private method that checks if file exists and is valid.
        
        Args:
            files (ImmutableMultiDict[str, FileStorage]): Files from the request form-data

        Returns:
            Tuple[any, int]: First value to return is the result of the image identification, second and last value to return is the status code.
        """
        
        # Check if 'image' key is in the form-data
        if "image" not in files:
            return jsonify(msg="No file found in request", status=400), 400

        image = files["image"]
        
        # Check if filename is not empty (this the same of been checking if form-data name is setted without value)
        if image.filename == "":
            return jsonify(msg="No selected file", status=400), 400
        
        # Check if fileextension is valid
        if not image.filename.endswith((".jpg", ".png", "jpeg")):
            return jsonify(msg="Not valid image", status=400), 400
        
        return jsonify(msg="Working", status=200), 200
    
    def process_images_handler(self, files: ImmutableMultiDict[str, FileStorage]) -> Tuple[any, int]:
        """Handler for the post route that processes the image to identify the number in it.

        Args:
            files (ImmutableMultiDict[str, FileStorage]): Files from the request form-data

        Returns:
            Tuple[any, int]: First value to return is the result of the image identification, second and last value to return is the status code.
        """
        
        try:
            json_msg, status = self.__check_valid_image(files)
            if status != 200:
                return json_msg, status

            # Get image from form-data
            image = files["image"]
            
            # Get image dimensions
            image_bytes = BytesIO(image.stream.read())
            img = Image.open(image_bytes)
            
            number_predicted = ImageService.indentify_number(img)
            
            n_predicted = int(number_predicted[0])
            probability = float(number_predicted[1])
        
            response = { "prediction": { "n_predicted": n_predicted, "probability": probability }, "status": 200 }
        
            return jsonify(response), status
        except Exception as e:
            return jsonify(msg="An error occurred", error=str(e), status=500), 500
        
    def get_weights(self):
        try:
            return jsonify(weights=ImageService.get_weights(), status=200), 200
        except Exception as e:
            return str(e), 500