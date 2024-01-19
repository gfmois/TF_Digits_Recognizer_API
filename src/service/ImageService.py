from ..models.mobileNetModel import modelInstance

class ImageService:
    @staticmethod
    def indentify_number(image):
        try:
            return modelInstance.indentify_number(image)
        except Exception as e:
            return e
        
    @staticmethod
    def get_weights():
        return modelInstance.check_instance()