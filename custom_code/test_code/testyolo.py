#!/usr/bin/env conda run -n yolov5 python
from concurrent.futures import wait
import torch
from PIL import Image
import numpy as np
import pandas
import sys, time
sys.path.append('C:/Users/pablo/source/repos/pytorch-cnn-visualizations/visualization_code')
from misc_functions import preprocess_image

def prepare_XAI(input):
    """
        Gets used variables for almost all visualizations, like the image, model etc.

    Args:
        example_index (int): Image id to use from examples

    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Read image
    original_image = Image.open(input).convert('RGB')
    # Process image
    prep_img = preprocess_image(original_image)
    return (original_image,
            prep_img)

if __name__ == '__main__':

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Define model
    pretrained_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    # Images
    img_paths = ['./input_images/carla_input/1.png','./input_images/carla_input/2.png']  # batch of images
    # pre-processing is unnecessary for YOLOv5
    for path in img_paths:
        original_image, preprocessed_image, model = prepare_XAI(path)
        # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        if torch.cuda.is_available():
            print("Torch cuda is available for the model")
            # preprocessed_image = preprocessed_image.to('cuda')
            model.to('cuda')
        else:
            print("Torch cuda is not available for the model")
        # Inference
            results = model(original_image)

    # Results
        results.print()
        results.show()  # or .save()        
        results.xyxy[0]  # img1 predictions (tensor)
        results.pandas().xyxy[0]  # img1 predictions (pandas)
        time.sleep(1)