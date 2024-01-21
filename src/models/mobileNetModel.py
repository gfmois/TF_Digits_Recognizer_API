import numpy as np
from PIL import Image
from typing import Tuple

from sklearn.datasets import load_digits
from skimage.transform import resize

from keras import Model
from keras.preprocessing import image
from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions


def get_data() -> Tuple[np.ndarray, np.ndarray]:
    data = load_digits()
    
    x = data.images
    y = data.target
    
    return x, y

_model = MobileNetV2(
    weights="imagenet",
    include_top=False
)

class MobileNetModel:
    def __init__(self, model_instance: Model) -> None:
        self.model = model_instance
        
    def __set_model(self, model: Model) -> None:
        self.model = model
        
    def transfer_learning(self, x: np.ndarray, y: np.ndarray,
                          resize_shape = (224, 224), output_neurons = 10, output_activation = "softmax", 
                          compile_loss = "sparse_categorical_crossentropy", compile_optimizer = "adam", 
                          compile_metrics = ["accuracy"], fit_epochs = 20, fit_verbose = 2):
        """
        Realizes the learning transfer to pretrained model.
        
        Args:
            x (np.ndarray): Input Data
            y (np.ndarry): Labels
            output_neurons (int, optional): Number of neurons in the output layer. Default is 10.
            output_activation (str, optional): Activation function for the output layer. Default is "softmax".
            compile_loss (str, optional): Loss function for model compilation. Default is "sparse_categorical_crossentropy".
            compile_optimizer (str, optional): Optimizer for model compilation. Default is "adam".
            compile_metrics (list, optional): Metrics for model compilation. Default is ["accuracy"].
            epochs (int, optional): Number of training epochs. Default is 20.
            verbose (int, optional): Verbosity level during training. Default is 2.
            
        Returns:
                Tuple[Model, dict]: Trained model and training history.
        """
        
        # Resize x images to selected shape, by default (224, 224)
        x_resized = np.array([resize(im, resize_shape, order=1, anti_aliasing=True) for im in x])
        
        # Add RGB Channel to the resized x's
        x_resized_rgb = np.stack((x_resized,) * 3, axis=-1)
        
        # GlobalAveragePooling2D layer to reduce spatial dimensions and summarize extracted features
        _x = GlobalAveragePooling2D()(self.model.output)
        
        # Dense layer added with specified number of neurons and activation function
        output_layer = Dense(output_neurons, activation=output_activation)(_x)
        
        # New model created by connecting the pre-trained model input with the new dense output layer
        model = Model(inputs=self.model.input, outputs=output_layer)
        
        # Freeze all layers except the new dense output layer
        for layer in model.layers[:-1]:
            layer.trainable = False
            
        # Compile the model
        model.compile(
            loss=compile_loss,
            optimizer=compile_optimizer,
            metrics=compile_metrics
        )
        
        # Train the model with the provided data
        model.fit(x=x_resized_rgb, y=y, epochs=fit_epochs, verbose=fit_verbose)
        
        # Set class model to new model
        self.__set_model(model)
        
        # Return trained model and the history of the training
        return model, model.history
    
    def indentify_number(self, img: Image.Image):
        """
        Receives an Image and with the MobileNetV2 model tryies to identifies it.
        
        Args:
            img (Image): Receives an Image to identify with the model MobileNetV2

        Returns:
            class_label (str): String with the value of the prediction
            probability (float): Probability of the tag to been the image 
        """
        
        img_array = self.__transform_img_to_array(img=img, shape=(224, 224))
        
        # Expand image to take the correct shape (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Preprocess image to be compatible with MobileNetV2 model
        img_array = preprocess_input(img_array)
        
        # Make the prediction
        predictions = self.model.predict(img_array)
        
        # Decode prediction
        decoded_prediction = decode_predictions(predictions)
        
        top_prediction = decoded_prediction[0][0]
        _, class_label, probability = top_prediction
       
        return class_label, probability
    
    def __transform_img_to_array(shape: dict = (224, 224), img: Image.Image | None = None) -> np.ndarray[any]:
        if img is  None:
            raise Exception("Arg 'img' cannot be None")
        
        # Resize image to the selected shape, by default (224, 224)
        img = img.resize(shape)
        
        # Transform input_image to numpy's array
        img_array = image.img_to_array(img)
        
        return img_array
        
    
    def check_instance(self):
        return self.model.get_weights()
    

x, y = get_data()
modelInstance = MobileNetModel(_model)

# TODO: If the model is already trained, don't train it again
modelInstance.transfer_learning(x, y)
modelInstance.indentify_number(x[0])