"""
@author: Pablo Roca - github.com/RocaPiedra
@original author: Utku Ozbulak - github.com/utkuozbulak
GradCAM visualizer for updated input options, including continuous feeds
"""
from time import sleep
from PIL import Image
from cv2 import *
import numpy as np
import torch
import cv2
import sys

from misc_functions import save_class_activation_images, apply_colormap_on_image
from roc_functions import *

try:
    import pygame
    import pygame.camera
    import pygame.image
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

sys.path.append('..')

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
            if(parameters.visualize):print(f'GPU acceleration is available')
        else:
            print(f'GPU acceleration is NOT available')
        if target_layer is not None:
            self.target_layer = target_layer
        else:
            self.target_layer = 11 #ReLU for Alexnet

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
                    if parameters.visualize: print("hook layer (target layer): ", module)
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
        self.method_name = 'GradCAM'
        self.class_name = '' # Variable that stores in object the detected class
        self.class_score = 0 # Variable that stores in object the score of the detected class
        # pygame initialization
        pygame.init()
        pygame.font.init() #for fonts rendering
        pygame.camera.init()
        cameras = pygame.camera.list_cameras()

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

    def visualization_pipeline(self, raw_data, sensor_platform=None, visualize_pipeline=True, visualize_original=False, color_palette=None):
        if sensor_platform is not None:
            frame = sensor_platform.carla_to_cv(raw_data)
        else:
            frame = raw_data

        if os.name == 'nt':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #error in linux
        
        if visualize_original:
            cv2.imshow('Input',frame)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print(f'Closing {cam.method_name}, shutting down application...')
                cv2.destroyAllWindows()
                exit()

        if color_palette is None:
            color_palette = 'hsv'
        
        if type(frame) != Image.Image:
            try:
                frame = Image.fromarray(frame)
            except Exception as e:
                print("could not transform PIL_img to a PIL Image object. Please check input.")
    
        prep_img = preprocess_image(frame)
        cam = self.generate_cam(prep_img)
        # Show mask
        _, heatmap_on_image = apply_colormap_on_image(frame, cam, color_palette)
        print(f'heatmap on image shape and type {np.shape(heatmap_on_image)} - {type(heatmap_on_image)}')
        cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
        print(f'cv2 heatmap on image shape and type {np.shape(cv2_heatmap_on_image)} - {type(cv2_heatmap_on_image)}')
        
        if visualize_pipeline:
            text = 'Class: ' + self.class_name + '\n' + 'Score: ' + str(np.round(self.class_score.detach().numpy(), 3))
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

        return cv2_heatmap_on_image

    def pygame_visualization_pipeline(self, raw_data, sensor_platform=None, visualize_pipeline=True, visualize_original=False, color_palette=None):
        if sensor_platform is not None:
            frame = sensor_platform.carla_to_cv(raw_data)
        else:
            frame = raw_data

        if os.name == 'nt':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #error in linux
        
        if visualize_original:
            cv2.imshow('Input',frame)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print(f'Closing {cam.method_name}, shutting down application...')
                cv2.destroyAllWindows()
                exit()

        if color_palette is None:
            color_palette = 'hsv'
        
        if type(frame) != Image.Image:
            try:
                frame = Image.fromarray(frame)
            except Exception as e:
                print("could not transform PIL_img to a PIL Image object. Please check input.")
    
        prep_img = preprocess_image(frame)
        cam = self.generate_cam(prep_img)
        # Show mask
        _, heatmap_on_image = apply_colormap_on_image(frame, cam, color_palette)
        print(f'heatmap on image shape and type {np.shape(heatmap_on_image)} - {type(heatmap_on_image)}')
        cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
        print(f'cv2 heatmap on image shape and type {np.shape(cv2_heatmap_on_image)} - {type(cv2_heatmap_on_image)}')
        
        if visualize_pipeline:
            text = 'Class: ' + self.class_name + '\n' + 'Score: ' + str(np.round(self.class_score.detach().numpy(), 3))
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

        return cv2_heatmap_on_image
