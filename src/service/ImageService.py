from ..models.DigitDetectorModel import modelInstance

class ImageService:
    @staticmethod
    def indentify_number(image):
        try:
            predict = modelInstance.predict(image)
            return predict
        except Exception as e:
            return e
        
    @staticmethod
    def get_weights():
        return modelInstance.check_instance()