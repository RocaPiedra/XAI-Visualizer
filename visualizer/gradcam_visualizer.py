"""
@author: Pablo Roca - github.com/RocaPiedra
@original author: Utku Ozbulak - github.com/utkuozbulak
"""
from time import sleep
from PIL import Image
from cv2 import *
import numpy as np
import torch
import ntpath # to obtain the filename in a full path
import os
import cv2
import sys

from misc_functions import save_class_activation_images, apply_colormap_on_image
from roc_functions import *

# for p in sys.path: print(f'1:{p}')
sys.path.append('..')
# for p in sys.path: print(f'2:{p}')

from carlacomms.carla_sensor_platform import sensor_platform
# code options
import parameters

prev_class = None

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        if torch.cuda.is_available() and parameters.sendToGPU:
            self.model.to('cuda')
        else:
            print(f'GPU acceleration is NOT available')
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        gpu = torch.device('cuda')

        if not x.is_cuda and parameters.sendToGPU:
            x = x.to(gpu)

        if self.model.__class__.__name__ == 'ResNet':
            for module_pos, module in self.model._modules.items():
                if parameters.visualize: print(f'*\n{module_pos}: {module}')
                if module_pos == "avgpool":
                    if parameters.visualize: print(f'*\nNext module is average pooling -> size is {module.output_size}; x size is: {x.size()}')
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
                    x = module(x)
                    if parameters.visualize: print(f'*\navgpool output size is: {x.size()}*\nhook layer for ResNet: {module}')
                    return conv_output, x # For ResNet after Avg Pool there is FC layer, skip here to avoid doing FC twice
                
                else:
                    if parameters.visualize: print(f"*\nforward pass in layer {module_pos} for ResNet")
                    x = module(x)

        else:
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
                    if parameters.visualize: print("hook layer: ", module)
                else:
                    if parameters.visualize: print("forward pass in layer: ", module)
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the ResNet model https://github.com/utkuozbulak/pytorch-cnn-visualizations/issues/50
        if self.model.__class__.__name__ == 'ResNet':
            # Forward pass on the convolutions
            if not x.is_cuda and parameters.sendToGPU:
                x = x.to('cuda')

            conv_output, x = self.forward_pass_on_convolutions(x)
            x = x.cpu() # return copy to CPU to use numpy
            x = x.reshape(x.size(0), -1)  # Flatten view for alexnet, reshape for resnet
            if not x.is_cuda and parameters.sendToGPU: # for GPU forward pass on classifier
                x = x.to('cuda')
            # Forward pass on the classifier
            x = self.model.fc(x)

        else:
            # Forward pass on the convolutions
            conv_output, x = self.forward_pass_on_convolutions(x)
            x = x.view(x.size(0), -1)  # Flatten
            # Forward pass on the classifier
            x = self.model.classifier(x)

        return conv_output, x

class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)
        # get the dictionary to obtain text results
        self.imagenet2txt = get_imagenet_dictionary()

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
                print(f'Last detected class is: {self.imagenet2txt[target_class]} | score: {model_output[0][target_class]}')
                prev_class = target_class

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        if self.model.__class__.__name__ == 'ResNet':
            self.model.zero_grad()
        else:
            self.model.features.zero_grad()
            self.model.classifier.zero_grad()
        
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        gradients = self.extractor.gradients.cpu() # return copy to CPU to use numpy
        guided_gradients = gradients.data.numpy()[0]
        # Get convolution outputs
        conv_output = conv_output.cpu() # return copy to CPU to use numpy
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Have a look at issue #11 to check why the above is np.ones and not np.zeros
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam

    def visualization_pipeline(self, raw_data, sensor_platform, visualize_pipeline=False):
        frame = sensor_platform.carla_to_cv(raw_data)
        original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prep_img = preprocess_image(original_image)
        cam = self.generate_cam(prep_img)
        # Show mask
        _, heatmap_on_image = apply_colormap_on_image(original_image, cam, 'hsv')
        cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
        if visualize_pipeline:
            cv2.imshow("GradCAM: Front Camera",cv2_heatmap_on_image)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print('Closing simulator camera, shutting down application...')
                cv2.destroyAllWindows()
                del self
                parameters.activate_deleter = True
                return parameters.activate_deleter
        return cv2_heatmap_on_image
