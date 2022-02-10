"""
@author: Pablo Roca - github.com/RocaPiedra
@reference: Utku Ozbulak - github.com/utkuozbulak
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

from misc_functions import save_class_activation_images, preprocess_image, apply_colormap_on_image
from roc_functions import get_image_path, choose_model, get_class_name_imagenet, get_imagenet_dictionary

# code options
visualize = False
sendToGPU = True
prev_class = None

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        if torch.cuda.is_available() and sendToGPU:
            if visualize: print('CUDA ENABLED in CamExtractor')
            self.model.to('cuda')
        else:
            print(f'Torch cuda is NOT available')
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

        if not x.is_cuda and sendToGPU:
            x = x.to(gpu)
            if visualize: print(f'in forward pass on convolutions x is in cuda -> {x.is_cuda}')

        if self.model.__class__.__name__ == 'ResNet':
            for module_pos, module in self.model._modules.items():
                print(f'*\n{module_pos}: {module}')
                # x = module(x)
                if module_pos == "avgpool":
                    print(f'*\nNext module is average pooling -> size is {module.output_size}; x size is: {x.size()}')
                    if module.output_size != (1, 512):
                        print(f'Size is not correct, change to (1, 512)')
                        module = torch.nn.AdaptiveAvgPool2d((1,1))
                        print(f'*\nNew average pooling module -> size is {module.output_size}')
                    # print(dir(module))
                    x.register_hook(self.save_gradient)
                    print(f'*\nAfter registering hook -> size is {module.output_size}')
                    conv_output = x  # Save the convolution output on that layer
                    x = module(x)
                    print(f'*\navgpool output size is: {x.size()}')
                    if visualize: print("hook layer for ResNet: ", module)
                else:
                    x = module(x)
                    if visualize: print("forward pass in layer for ResNet: ", module)
        else:
            for module_pos, module in self.model.features._modules.items():
                x = module(x)  # Forward
                if int(module_pos) == self.target_layer:
                    x.register_hook(self.save_gradient)
                    conv_output = x  # Save the convolution output on that layer
                    if visualize: print("hook layer: ", module)
                else:
                    if visualize: print("forward pass in layer: ", module)
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the ResNet model https://github.com/utkuozbulak/pytorch-cnn-visualizations/issues/50
        if self.model.__class__.__name__ == 'ResNet':
            # Forward pass on the convolutions
            if not x.is_cuda and sendToGPU:
                if visualize: print('**\nx in forward pass to GPU\n**')
                x = x.to('cuda')
                if visualize: print(f'in forward pass x is in cuda {x.is_cuda}')
            conv_output, x = self.forward_pass_on_convolutions(x)
            x = torch.transpose(x, 1, 0)
            x = x.cpu() # return copy to CPU to use numpy
            # x = x.reshape(x.size(0), -1)  # Flatten view for alexnet, reshape for resnet
            x = x.view(x.size(0),-1)
            if not x.is_cuda and sendToGPU: # for GPU forward pass on classifier
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
                print(f'Last detected class is: {imagenet_dictionary[target_class]}')
                prev_class = target_class

        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
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
        # ^ I am extremely unhappy with this line. Originally resizing was done in cv2 which
        # supports resizing numpy matrices with antialiasing, however,
        # when I moved the repository to PIL, this option was out of the window.
        # So, in order to use resizing with ANTIALIAS feature of PIL,
        # I briefly convert matrix to PIL image and then back.
        # If there is a more beautiful way, do not hesitate to send a PR.

        # You can also use the code below instead of the code line above, suggested by @ ptschandl
        # from scipy.ndimage.interpolation import zoom
        # cam = zoom(cam, np.array(input_image[0].shape[1:])/np.array(cam.shape))
        return cam


if __name__ == '__main__':

    #Use input Arguments
    input_arguments = sys.argv
    print(input_arguments)

    images_processed = int(0) #debug
    imagenet_dictionary = get_imagenet_dictionary()
    # Choose model
    if len(input_arguments) >= 2:
        if len(input_arguments[1])==1:
            pretrained_model = choose_model(input_arguments[1], modelname = None)
        elif len(input_arguments[1])>1:
            pretrained_model = choose_model(0, input_arguments[1])
    else:
        pretrained_model = choose_model()
    if torch.cuda.is_available() and sendToGPU:
        if visualize: print('CUDA ENABLED in main')
        pretrained_model.to('cuda')
        if visualize: print(f'is model loaded to gpu? -> {next(pretrained_model.parameters()).is_cuda}')
    elif not sendToGPU:
        print('Using CPU') 
    else:
        print('CUDA NOT ENABLED in main, exit')
        exit()
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)

    # Choose input option
    if len(input_arguments) >= 3:    
        if int(input_arguments[2]) in (1,2,3):
            option = int(input_arguments[2])
    else:
        option = int(input('What type of input do you want: \n1.Webcam\n2.Use a path to open images\n3.Use a path to open video\n'))
    if option == 1:
        print('Webcam selected as input')
        frame_counter = 0
        cap = cv2.VideoCapture(0)   # /dev/video0
        while True:
            ret, frame = cap.read()
            frame_counter += 1
            if not ret:
                break
            original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('Webcam',frame)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                break
            prep_img = preprocess_image(original_image, sendToGPU)
            file_name_to_export = f'webcam_{frame_counter}'
            
            if not prep_img.is_cuda:
                if visualize: print('**\nPreprocessed image to GPU\n**')
                prep_img = prep_img.to('cuda')
                if visualize: print(f'in main prep_img is in cuda {prep_img.is_cuda}')
            cam = grad_cam.generate_cam(prep_img)
            # Show mask
            heatmap, heatmap_on_image = apply_colormap_on_image(original_image, cam, 'hsv')
            cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
            cv2.imshow('GradCam',cv2_heatmap_on_image)

        cap.release()
        cv2.destroyAllWindows()

    elif option == 2:
        print('Image selected as input')
        path = '../input_images/carla_input/'
        image_paths = get_image_path(path,None)
        # do a function for target class
        for paths in image_paths:
            original_image = Image.open(paths).convert('RGB')
            prep_img = preprocess_image(original_image)
            # take the name of the file without extension:
            file_name_to_export = os.path.splitext(ntpath.basename(paths))[0]
            # Generate cam mask
            cam = grad_cam.generate_cam(prep_img)
            # Save mask
            save_class_activation_images(original_image, cam, file_name_to_export)
            print('Grad cam completed for image:',file_name_to_export)
            images_processed+=1

    if option == 3:
        print('Video selected as input')
        frame_counter = 0
        video_path = '../input_images/video_input/Road-traffic.mp4'
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            frame_counter += 1
            if not ret:
                break
            original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('Webcam',frame)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                break
            prep_img = preprocess_image(original_image)
            file_name_to_export = f'webcam_{frame_counter}'
            cam = grad_cam.generate_cam(prep_img)
            # Show mask
            heatmap, heatmap_on_image = apply_colormap_on_image(original_image, cam, 'hsv')
            cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
            cv2.imshow('GradCam',cv2_heatmap_on_image)

        cap.release()
        cv2.destroyAllWindows()

    else:
        print('Wrong option, shutting down application...')
        exit()

    elapsed = '*unknown*'
    print(f'A total of {images_processed} images received gradcam in {elapsed} seconds')