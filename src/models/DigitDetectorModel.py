import random
import tensorflow as tf
import os

import numpy as np
from PIL import Image
from typing import Tuple

from ..utils import utils

from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from keras.models import load_model, Model
from keras.src.callbacks import History
from keras.layers import Input, Conv2D, Dense, GlobalAveragePooling2D, MaxPooling2D


class DigitDetectorModel:
    def __init__(self, model_path: str | None = None) -> None:
        self.model = None
        self.x = None
        self.y = None
        
        self.x, self.y = self.get_data()
        self.y_class = LabelEncoder().fit_transform(self.y)
        
        if model_path is not None:
            self.load_model_from_h5(model_path)
        
    def get_data(self):
        try:
            data = load_digits()
            return data.images, data.target
        except Exception as e:
            raise Exception(e)
        
    def __set_model(self, model: Model) -> None:
        """Changes the model value to new Model.

        Args:
            model (Model): New Keras Model Type.
        """
        self.model = model
        
    def load_model_from_h5(self, path_to_h5: str):
        """
        Load pretrained model in .h5 extension to use it in the app.

        Args:
            path_to_h5 (str): The path to the .h5 file.
        """
        try:
            self.__set_model(load_model(path_to_h5))
        except Exception as e:
            raise Exception(e)
    
    def save_model(self, name_of_model: str = "digit_recognizer.h5"):
        """Saves the current model to .h5 extension.

        Args:
            name_of_model (str, optional): Name of the file where the model weights will be saved. Defaults to "digits_classificator.h5".
        """
        try:
            if not name_of_model.endswith(".h5"):
                name_of_model += ".h5"
            
            self.model.save(name_of_model)
        except Exception as e:
            raise Exception(e)
        
    def get_model_architecture(self, input_shape = (224, 224, 3)) -> Model:
        """Constructs the model architecture, this method needs to improve to make custom architectures

        Args:
            input_shape (tuple, optional): _description_. Defaults to (224, 224, 3).

        Returns:
            Model: Model with the architecture formed
        """
        input_layer = Input(input_shape)

        conv2d_1_0 = Conv2D(filters=32, padding="same", kernel_size=(3, 3), activation="relu")(input_layer)
        conv2d_1_1 = Conv2D(filters=32, padding="same", kernel_size=(3, 3), activation="relu")(conv2d_1_0)
        maxpooling_1 = MaxPooling2D()(conv2d_1_1)

        conv2d_2 = Conv2D(filters=64, padding="same", strides=(2, 2), kernel_size=(5, 5), activation="relu")(maxpooling_1)
        maxpooling_2 = MaxPooling2D()(conv2d_2)

        conv2d_3_0 = Conv2D(filters=128, kernel_size=(7, 7), activation="relu")(maxpooling_2)
        conv2d_3_1 = Conv2D(filters=128, kernel_size=(7, 7), activation="relu")(conv2d_3_0)
        gavg_pooling = GlobalAveragePooling2D()(conv2d_3_1)

        d0 = Dense(units=64, activation="relu")(gavg_pooling)
        d1 = Dense(units=32, activation="relu")(d0)
        exit_layer = Dense(units=10, activation="sigmoid")(d1)
        
        model = Model(inputs=[input_layer], outputs=[exit_layer])
        
        return model
        
    def train_model(self, seed = 17, loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"]) -> History:
        """
        Train model with custom seed, loss, optimizer and metrics.

        Args:
            seed (int, optional): Seed setted to model fit. Defaults to 17.
            loss (str, optional): loss function. Defaults to "sparse_categorical_crossentropy".
            optimizer (str, optional): optimizer. Defaults to "adam".
            metrics (list, optional): metrics to get. Defaults to ["accuracy"].

        Returns:
            History: History of the model fit
        """
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)
        
        y_class = LabelEncoder().fit_transform(self.y)
        n_class = len(np.unique(y_class))
        
        X_train, X_test, y_train, y_test = train_test_split(self.x, y_class, test_size=0.2, random_state=seed)
        
        X_train, X_test = utils.resize_imgs(X_train, color_upgrade=50), utils.resize_imgs(X_test, color_upgrade=50)
        
        shape = X_train[0].shape
        
        model = self.get_model_architecture(input_shape=shape)
        
        model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics    
        )
        
        history: History = model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test), batch_size=32)
        self.__set_model(model)
        
        return history
    
    def predict(self, img: Image.Image) -> Tuple[int, float]:
        """
        Receives an Image and with the MobileNetV2 model tryies to identifies it.
        
        Args:
            img (Image): Receives an Image to identify with the model MobileNetV2

        Returns:
            class_label (str): String with the value of the prediction
            probability (float): Probability of the tag to been the image 
        """
        try:
            img_array = np.array(img)
            resized_img = utils.resize_imgs([img_array], color_upgrade=10, from_file=True)[0]
            img_to_predict = np.expand_dims(resized_img, axis=0)
            predicted_result = self.model.predict(img_to_predict)[0]
            
            response = {}
            
            n_predicted = np.max(predicted_result)
            index_n_predicted = np.where(predicted_result == n_predicted)[0][0]
            y_predicted = self.y_class[index_n_predicted]
            
            response["n_predicted"] = y_predicted
            response["probability"] = n_predicted
            
            return y_predicted, n_predicted
        except Exception as e:
            return { "error": str(e) }
    

modelInstance = DigitDetectorModel(model_path=f"{os.getcwd()}/examples/digit_recognizer_good.h5")