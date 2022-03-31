"""
@author: Pablo Roca - github.com/RocaPiedra
@original author: Utku Ozbulak - github.com/utkuozbulak
"""
from time import sleep
from PIL import Image
from cv2 import *
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import sys

from misc_functions import apply_colormap_on_image
from roc_functions import *

sys.path.append('..')

from carlacomms.carla_sensor_platform import sensor_platform
# code options
import parameters

prev_class = None

class ScoreCamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        if torch.cuda.is_available() and parameters.sendToGPU:
            self.model.to('cuda')
            print(f'is model in cuda: {next(model.parameters()).is_cuda}')
        else:
            print(f'GPU acceleration is NOT available')
        if target_layer is not None:
            self.target_layer = target_layer
        else:
            self.target_layer = 11 #ReLU for Alexnet

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        gpu = torch.device('cuda')

        if not x.is_cuda and parameters.sendToGPU:
            x = x.to(gpu)
            print(f'sent input to cuda, worked? {x.is_cuda}')
        
        if self.model.__class__.__name__ == 'ResNet':
            for module_pos, module in self.model._modules.items():
                # if module_pos == self.target_layer:
                if module_pos == "avgpool":
                    x = module(x)  # Forward
                    conv_output = x  # Save the convolution output on that layer
                    if parameters.visualize: print("target layer: ", module)
                    return conv_output, x # For ResNet after Avg Pool there is FC layer, skip here to avoid doing FC twice
                else:
                    x = module(x)  # Forward
        else:
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    conv_output = x  # Save the convolution output on that layer
                    if parameters.visualize: print("target layer: ", module)
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        if not x.is_cuda and parameters.sendToGPU:
            x = x.to('cuda')
        # Forward pass on the convolutions
        if self.model.__class__.__name__ == 'ResNet':
            conv_output, x = self.forward_pass_on_convolutions(x)
            x = x.cpu() # return copy to CPU to use numpy
            x = x.reshape(x.size(0), -1)  # Flatten view for alexnet, reshape for resnet
            if not x.is_cuda and parameters.sendToGPU: # for GPU forward pass on classifier
                x = x.to('cuda')
            # Forward pass on the classifier
            x = self.model.fc(x)
        
        else:
            conv_output, x = self.forward_pass_on_convolutions(x)
            x = x.view(x.size(0), -1)  # Flatten
            if not x.is_cuda and parameters.sendToGPU: # for GPU forward pass on classifier
                x = x.to('cuda')
            # Forward pass on the classifier
            x = self.model.classifier(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = ScoreCamExtractor(self.model, target_layer)
        # get the dictionary to obtain text results
        self.imagenet2txt = get_imagenet_dictionary()
        self.method_name = 'ScoreCAM'
        self.class_name = 'Empty' # Variable that stores in object the detected class
        self.class_score = 0 # Variable that stores in object the score of the detected class

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        global prev_class

        if target_class is None:
            model_output = model_output.cpu() # return copy to CPU to use numpy
            target_class = np.argmax(model_output.data.numpy())
            if target_class != prev_class:
                self.class_name = self.imagenet2txt[target_class]
                self.class_score = model_output[0][target_class]
                print(f'Last detected class is: {self.class_name} | score: {self.class_score}')
                prev_class = target_class

        # Target for backprop
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False).to('cuda')
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            if norm_saliency_map.is_cuda != input_image.is_cuda:
                input_image = input_image.to('cuda')
                norm_saliency_map = norm_saliency_map.to('cuda')
            
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1).to('cuda')[0][target_class]
            if w.is_cuda:
                w = w.cpu()
            if target.is_cuda:
                target = target.cpu()
                
            cam += w.data.numpy() * target[i, :, :].data.numpy()
        cam = np.maximum(cam, 0)
        # print(f'cam: {cam}\n min cam: {np.min(cam)}\n max cam: {np.max(cam)}\n')
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam


    def visualization_pipeline(self, raw_data, sensor_platform=None, visualize_pipeline=False, visualize_original=False):
        
        if sensor_platform is not None:
            frame = sensor_platform.carla_to_cv(raw_data)
        else:
            frame = raw_data

        original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if visualize_original:
            cv2.imshow('Input',original_image)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print(f'Closing {cam.method_name}, shutting down application...')
                cv2.destroyAllWindows()
                exit()

        prep_img = preprocess_image(original_image)
        cam = self.generate_cam(prep_img)
        # Show mask
        _, heatmap_on_image = apply_colormap_on_image(original_image, cam, 'hsv')
        cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
        
        if visualize_pipeline:
            try:
                text = 'Class: ' + self.class_name + '\n' + 'Score: ' + str(np.round(self.class_score.detach().numpy(), 3))
            except:
                text = 'Class: ' + self.class_name + '\n' + 'Score: ' + str(np.round(self.class_score.numpy(), 3))
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 255, 255)
            thickness = 1
            y0, dy = 12, 14
            for i, line in enumerate(text.split('\n')):
                y = y0 + i*dy
                cv2_heatmap_on_image = cv2.putText(cv2_heatmap_on_image, line, (10,y),
                                font, fontScale, color, thickness, cv2.LINE_AA, False)
            cv2.imshow(f"{self.method_name}",cv2_heatmap_on_image)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print('Closing simulator camera, shutting down application...')
                cv2.destroyAllWindows()
                del self
                parameters.activate_deleter = True
                return parameters.activate_deleter
                
        return cv2_heatmap_on_image

        
    def old_visualization_pipeline(self, raw_data, sensor_platform, visualize_pipeline=False):
        frame = sensor_platform.carla_to_cv(raw_data)
        original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prep_img = preprocess_image(original_image)
        cam = self.generate_cam(prep_img)
        # Show mask
        _, heatmap_on_image = apply_colormap_on_image(original_image, cam, 'hsv')
        cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
        if visualize_pipeline:
            cv2.imshow("ScoreCAM: Front Camera",cv2_heatmap_on_image)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print('Closing simulator camera, shutting down application...')
                cv2.destroyAllWindows()
                del self
                parameters.activate_deleter = True
                return parameters.activate_deleter
        return cv2_heatmap_on_image
