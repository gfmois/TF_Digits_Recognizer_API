from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras import Model
from keras.preprocessing import image
import numpy as np
from PIL import Image

_model = MobileNetV2(
    weights="imagenet"
)

class MobileNetModel:
    def __init__(self, model_instance: Model) -> None:
        self.model = model_instance
    
    def indentify_number(self, img: Image):
        """_summary_

        Args:
            img (Image): Receives an Image to identify with the model MobileNetV2

        Returns:
            class_label (str): String with the value of the prediction
            probability (float): Probability of the tag to been the image 
        """
        
        # Resize image to the model shape (224, 224)
        img = img.resize((224, 224))
        
        # Transform input_image to numpy's array
        img_array = image.img_to_array(img)
        
        # Expand image to take the correct shape (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess image to be compatible with MobileNetV2 model
        img_array = preprocess_input(img_array)
        
        # Make the prediction
        predictions = self.model.predict(img_array)
        
        # Decode prediction
        decoded_prediction = decode_predictions(predictions)
        
        top_prediction = decode_predictions[0][0]
        _, class_label, probability = top_prediction
       
        return class_label, probability

    def check_instance(self):
        return self.model.get_weights()
    
modelInstance = MobileNetModel(_model)