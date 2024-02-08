import math
from typing import Tuple

from PIL import Image
import PIL.ImageOps as ImgOpts

import numpy as np
from keras.preprocessing import image
from skimage.transform import resize

class Colors:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

def img_to_array(shape = (224, 224), img: Image.Image | None = None):
    if img is None:
        raise Exception("Arg 'img' cannot be None.")
    
    img = img.resize(shape)
    
    img_array = image.img_to_array(img)
    
    return img_array

def add_padding_to_img(pil_img: Image.Image, top, bottom, left, right, color=(0, 0, 0)):    
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    
    result = Image.new("RGB", (new_width, new_height), color=color)
    result.paste(pil_img, (left, top))
    
    return result

def calculate_padding(initial_size: Tuple[int, int], target_size: Tuple[int, int] = (224, 224)) -> Tuple[int, int, int, int]:
        vertical_padding = max(0, math.ceil((target_size[0] - initial_size[0]) / 2))
        horizontal_padding = max(0, math.ceil((target_size[1] - initial_size[1]) / 2))
        return (vertical_padding, vertical_padding, horizontal_padding, horizontal_padding)

def resize_imgs(images: Tuple[Image.Image], new_shape = (220, 220), resize_mode = "constant",  color_upgrade = 10, color_filter = 5, from_file: bool = False):
    new_images = np.empty((len(images), 224, 224, 3))
    
    for i, img in enumerate(images):
        img = resize(img, new_shape, anti_aliasing=True, mode=resize_mode)
        
        if from_file:
            img[img != 0] = 255
        
        img = Image.fromarray((img).astype(np.uint8))
        img = add_padding_to_img(img, *calculate_padding(new_shape))
        
        new_data = []
        img = img.convert("RGB")
        
        for item in img.getdata():
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                new_data.append(Colors.BLACK)
            else:
                n_color = tuple([color * color_upgrade if color > color_filter else color for color in item])
                new_data.append(n_color)
                
        img.putdata(new_data)
        img = ImgOpts.invert(img)
        img_to_array = np.array(img)
        new_images[i] = img_to_array
        
    return new_images