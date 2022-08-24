"""

@author: Pablo Roca - github.com/RocaPiedra
"""
import os
import sys
import numpy as np
from PIL import Image

import pygame
from pygame.locals import *
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
sys.path.append('../visualizer')
from roc_functions import draw_text, surface_to_cam

import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, alexnet, vgg11, vgg19
import parameters

from parameters import BUTTON_COLOR, WHITE

from urllib.request import urlopen
from multiprocessing import Process
import roc_functions
import time
import gc

debug = False

class menu:
    def __init__(self, display, use_cuda = True):
        self.model = resnet18(pretrained=True)
        self.display = display
        self.use_cuda = use_cuda
        self.font = pygame.font.SysFont(None, 24)
        self.target_layers = self.select_target_layer()
        self.cam_name = None
        self.method_name = None
        self.model_name = 'ResNet'
        self.CAM_BUTTON_COLOR = BUTTON_COLOR
        self.MODEL_BUTTON_COLOR = BUTTON_COLOR
        self.click = False
        self.class_list = roc_functions.get_imagenet_dictionary(url=None) 
        self.cam = None
        self.classification_output = ''  
        if torch.cuda.is_available() and self.use_cuda:
            self.model.to('cuda')
            print("System is cuda ready")
            
        self.surface = None
        self.main_location = None
        w, h = pygame.display.get_surface().get_size()
        #location where second cam is plotted in the display
        self.compare_location = [int(2*w/3), int(h/2)]
        print(f"the compare location is {self.compare_location}")
     
        
    def select_cam(self, second_method = False):            
        method_selection = True
        x = 100
        y = 100
        dx = 0
        dy = 100
        num_options = 7
        positions = []
        for pos in range(num_options):
            positions.append([x+pos*dx, y+pos*dy])

        while method_selection:
            
            draw_text('Model Menu', self.font, (255, 255, 255), self.display, 20, 20)
            
            mx, my = pygame.mouse.get_pos()
            # To delimit the size of the button, in the future use value related to window res
            w, h = pygame.display.get_surface().get_size()
            button_width = 300
            button_height = 40
            
            grad_button = pygame.Rect(positions[0][0], positions[0][1], button_width, button_height)
            score_button = pygame.Rect(positions[1][0], positions[1][1], button_width, button_height)
            xgradcam_button = pygame.Rect(positions[2][0], positions[2][1], button_width, button_height)
            ablation_button = pygame.Rect(positions[3][0], positions[3][1], button_width, button_height)
            eigen_button = pygame.Rect(positions[4][0], positions[4][1], button_width, button_height)
            fullgrad_button = pygame.Rect(positions[5][0], positions[5][1], button_width, button_height)
            gradcampp_button = pygame.Rect(positions[6][0], positions[6][1], button_width, button_height)

            pygame.draw.rect(self.display, self.CAM_BUTTON_COLOR, grad_button)
            draw_text('GradCAM', self.font, (255, 255, 255), self.display, positions[0][0], positions[0][1]+button_height-15)
            pygame.draw.rect(self.display, self.CAM_BUTTON_COLOR, score_button)
            draw_text('ScoreCAM', self.font, (255, 255, 255), self.display, positions[1][0], positions[1][1]+button_height-15)
            pygame.draw.rect(self.display, self.CAM_BUTTON_COLOR, xgradcam_button)
            draw_text('XGradCAM', self.font, (255, 255, 255), self.display, positions[2][0], positions[2][1]+button_height-15)
            pygame.draw.rect(self.display, self.CAM_BUTTON_COLOR, ablation_button)
            draw_text('AblationCAM', self.font, (255, 255, 255), self.display, positions[3][0], positions[3][1]+button_height-15)
            pygame.draw.rect(self.display, self.CAM_BUTTON_COLOR, eigen_button)
            draw_text('EigenCAM', self.font, (255, 255, 255), self.display, positions[4][0], positions[4][1]+button_height-15)
            pygame.draw.rect(self.display, self.CAM_BUTTON_COLOR, fullgrad_button)
            draw_text('FullGrad', self.font, (255, 255, 255), self.display, positions[5][0], positions[5][1]+button_height-15)
            pygame.draw.rect(self.display, self.CAM_BUTTON_COLOR, gradcampp_button)
            draw_text('GradCAM++', self.font, (255, 255, 255), self.display, positions[6][0], positions[6][1]+button_height-15)

            if grad_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'GradCAM'
                    
            if score_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'ScoreCAM'
                    
            if ablation_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'AblationCAM'
                    
            if xgradcam_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'XGradCAM'
                    
            if eigen_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'EigenCAM'
                    
            if fullgrad_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'FullGrad'
                    
            if gradcampp_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'GradCAM++'
                    
            self.click = False
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.click = True
            pygame.display.update()
        if second_method is False:
            self.method_name = method_name
        return method_name    
    
    
    def load_cam_if(self, cuda_error = False, method_name=None):
        
        if not method_name:
            method_name = self.method_name
            
        if method_name == 'ScoreCAM':
            if not cuda_error:
                cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'AblationCAM':
            if not cuda_error:
                cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'XGradCAM':
            if not cuda_error:
                cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'EigenCAM':
            if not cuda_error:
                cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'FullGrad':
            if not cuda_error:
                cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'GradCAM++':
            if not cuda_error:
                cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if method_name == 'GradCAM':
            if not cuda_error:
                cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if cam_method:    
            return cam_method
        else:
            self.select_cam()
        
    # Compare methods while managing GPU memory usage to avoid errors
    def compare_new_method(self, img):
        new_method_name = self.select_cam(second_method = True)
        old_method_name = self.method_name
        new_cam_method = self.load_cam_if(method_name = new_method_name)
        class_name, class_score = self.run_cam(img, new_cam_method, self.compare_location, new_method_name)
        print(f"compared {old_method_name} to {new_method_name} \
            -> finished with output {class_name}|{class_score}%")
        time.sleep(10)
        # After execution it is necessary to free memory by deleting the second method
        del new_cam_method
        gc.collect()
        torch.cuda.empty_cache()
        # Reload initial method, deleted in load cam if to free GPU memory
        self.cam = self.load_cam_if(False, method_name = old_method_name)
        return class_name, class_score    
    
    
    def select_target_layer(self):
        # to implement for multiple models
        if self.model.__class__.__name__ == 'ResNet':
            self.target_layers = [self.model.layer4[-1]]
            print(f'Target Layer for {self.model.__class__.__name__} is:')
            
        elif self.model.__class__.__name__ == 'Alexnet':
            self.target_layers = [11]
            print(f'Target Layer for {self.model.__class__.__name__} is:')
            
        elif self.model.__class__.__name__ == 'VGG':
            self.target_layers = [self.model.features[-1]]
            print(f'Target Layer for {self.model.__class__.__name__} is:')
            
        elif self.model.__class__.__name__ == 'AutoShape':
            self.target_layers = [self.model.model.model.model[-2]]
            print(f'Target Layer for YOLOv5 is:')
            
        print(self.target_layers)    
        return self.target_layers
    
    
    def select_model(self):
        model_selection = True
        x = 100
        y = 100
        dx = 0
        dy = 100
        num_options = 4 
        positions = []
        for pos in range(num_options):
            positions.append([x+pos*dx, y+pos*dy])

        while model_selection:
            
            draw_text('model Menu', self.font, (255, 255, 255), self.display, 20, 20)
            
            mx, my = pygame.mouse.get_pos()
            # To delimit the size of the button, in the future use value related to window res
            w, h = pygame.display.get_surface().get_size()
            button_width = 300
            button_height = 40
            
            resnet_button = pygame.Rect(positions[0][0], positions[0][1], button_width, button_height)
            alexnet_button = pygame.Rect(positions[1][0], positions[1][1], button_width, button_height)
            third_button = pygame.Rect(positions[2][0], positions[2][1], button_width, button_height)
            fourth_button = pygame.Rect(positions[3][0], positions[3][1], button_width, button_height)

            pygame.draw.rect(self.display, BUTTON_COLOR,  resnet_button)
            draw_text('ResNet', self.font, (255, 255, 255), self.display, positions[0][0], positions[0][1]+button_height-15)
            pygame.draw.rect(self.display, BUTTON_COLOR, alexnet_button)
            draw_text('Alexnet', self.font, (255, 255, 255), self.display, positions[1][0], positions[1][1]+button_height-15)
            pygame.draw.rect(self.display, BUTTON_COLOR, third_button)
            draw_text('VGG', self.font, (255, 255, 255), self.display, positions[2][0], positions[2][1]+button_height-15)
            pygame.draw.rect(self.display, BUTTON_COLOR, fourth_button)
            draw_text('YOLOv5', self.font, (255, 255, 255), self.display, positions[3][0], positions[3][1]+button_height-15)

            if resnet_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'ResNet':
                        self.clear_memory()
                        self.model = resnet18(pretrained=True)
                        self.select_target_layer()
                        if self.cam:
                            try:
                                print('Reloading CAM method with new Model')
                                self.cam = self.load_cam_if()
                            except:
                                print('Some error ocurred, try loading cam to CPU')
                                self.cam = self.load_cam_if(True)
                        else:
                            print('CAM method has not been selected, press M to choose one')
                                
                        model_selection = False
                        model_name = 'ResNet'
                        if self.use_cuda:
                            self.model.to('cuda')
                            print(f"Selected model {model_name} is cuda ready")
                    else:
                        print(f'Model selected -> {self.model.__class__.__name__} was already loaded')
            
            if alexnet_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'Alexnet':
                        self.clear_memory()
                        self.model = alexnet(pretrained=True) 
                        self.select_target_layer()
                        if self.cam:
                            try:
                                print('Reloading CAM method with new Model')
                                self.cam = self.load_cam_if()
                            except:
                                print('Some error ocurred, try loading cam to CPU')
                                self.cam = self.load_cam_if(True)
                        else:
                            print('CAM method has not been selected, press M to choose one')
                            
                        model_selection = False
                        model_name = 'Alexnet'
                        if self.use_cuda:
                            self.model.to('cuda')
                            print(f"Selected model {model_name} is cuda ready")
                    else:
                        print(f'Model selected -> {self.model.__class__.__name__} was already loaded')
            
            if third_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'VGG':
                        self.clear_memory()
                        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
                        self.select_target_layer()
                        if self.cam:
                            try:
                                print('Reloading CAM method with new Model')
                                self.cam = self.load_cam_if()
                            except:
                                print('Some error ocurred, try loading cam to CPU')
                                self.cam = self.load_cam_if(True)
                        else:
                            print('CAM method has not been selected, press M to choose one')
                                
                        model_selection = False
                        model_name = 'VGG'
                        if self.use_cuda:
                            self.model.to('cuda')
                            print(f"Selected model {model_name} is cuda ready")
                    else:
                        print(f'Model selected -> {self.model.__class__.__name__} was already loaded')
                    
            if fourth_button.collidepoint((mx, my)):
                if self.click:
                    # not sure why, yolov5 returns as name AutoShape.
                    if self.model.__class__.__name__ != 'AutoShape':
                        self.clear_memory()
                        self.model =  torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                        self.select_target_layer()
                        if self.cam:
                            try:
                                print('Reloading CAM method with new Model')
                                self.cam = self.load_cam_if()
                            except:
                                print('Some error ocurred, try loading cam to CPU')
                                self.cam = self.load_cam_if(True)
                        else:
                            print('CAM method has not been selected, press M to choose one')        
                        model_selection = False
                        model_name = 'YOLOv5'
                        if self.use_cuda:
                            self.model.to('cuda')
                            print(f"Selected model {model_name} is cuda ready")
                    else:
                        print(f'Model selected -> YOLOv5 was already loaded')
            
            self.click = False
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                if event.type == MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.click = True
                        
            pygame.display.update()
        self.model_name = model_name
        return model_name

    #This option obtains the inference results from outside the cam method
    def get_detection(self, img):
        output = self.run_model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probabilities = probabilities.to('cpu')
        target_class = np.argmax(probabilities.detach().numpy())
        class_name = self.class_list[target_class]
        class_score = probabilities[target_class]
        if debug:
            print(f'target class is {target_class}')
            print(f"SINGLE DETECTION: {class_name} || {class_score*100}% ")
            
        return class_name, class_score
    
    
    def get_top_detections(self, input_image = None, probabilities = None, num_detections = 5):
        top_locations = np.argpartition(probabilities, -num_detections)[-num_detections:]
        ordered_locations = top_locations[np.argsort((-probabilities)[top_locations])]
        np.flip(ordered_locations)
        
        return ordered_locations
    
    
    def prob_calc_efficient(self, output):
        # The output has unnormalized scores. To get probabilities, run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probabilities = probabilities.to('cpu')
        target_class = np.argmax(probabilities.data.numpy())
        class_name = self.class_list[target_class]
        class_score = probabilities[target_class]
        return class_name, class_score.cpu().detach().numpy()
    
    
    def run_cam(self, img, cam_method = None, selected_location = None, new_method_name = None):
        if cam_method is None:
            cam_method = self.cam
        else:
            # try this to free GPU memory and avoid errors (cam must be instanced afterwards again!)
            del self.cam
            
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        # get the cam heat map in a pygame image
        surface, inf_outputs, cam_targets =  roc_functions.surface_to_cam(img, cam_method, self.use_cuda)
        print('time needed for visualization method creation :', time.time()-t0)
        
        t1 = time.time()
        class_name, class_score = self.prob_calc_efficient(inf_outputs)
        class_percentage = str(round(class_score*100,2))
        print('time needed for probabilities calculation:', time.time()-t1)
        
        if selected_location is None:
            self.surface = surface
            self.render()
            # self.render_text()
        else:
            score_string = f"Class detected: {class_name} with score: {class_percentage}%"
            self.render(selected_location, surface, score_string, new_method_name)
            # self.render_text()
        
        return class_name, class_percentage

    
    def render(self, selected_location = None, surface_to_plot = None, second_classification = None, new_method_name = None):
        if surface_to_plot is None:
            self.display.blit(self.surface, self.main_location)
            pygame.display.update() 
            self.text_render()

        elif selected_location == self.compare_location:
            print('plotting the second CAM output...')
            self.display.blit(surface_to_plot, selected_location)
            pygame.display.update()
            if second_classification is not None and new_method_name is not None:
                self.text_render(second_classification, new_method_name)


    def text_render(self, second_classification = None, second_method_name = None):
        
        if second_method_name is not None:
            description = f'Model: {self.model_name} Method: {second_method_name}'
        else:
            description = f'Model: {self.model_name} Method: {self.cam_name}'
            
        description_text = self.font.render(description, True, (255, 255, 255))
        
        if second_classification is None:
            score_output = self.font.render(self.classification_output, True, (255, 255, 255))
            loc = self.main_location
        else:
            print(f'second classification is {second_classification}')
            score_output = self.font.render(second_classification, True, (255, 255, 255))
            loc = self.compare_location
        
        score_loc = [loc[0], loc[1] + 20]
        self.display.blit(description_text, loc)
        self.display.blit(score_output, score_loc)    
        pygame.display.update()
    
    # img is a surface 
    def run_model(self, img):
        with torch.no_grad():
            preprocessed_image = pygame.surfarray.pixels3d(img)
            if debug:
                preprocess_pil = Image.fromarray(np.uint8(preprocessed_image))
                preprocess_pil.show()
                input("wait for user input to pass surface image")
                
            preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            preprocess_pil = Image.fromarray(np.uint8(preprocessed_image))
            if debug:
                preprocess_pil.show()
                input("wait for user input to pass preprocessed image")
                
            input_tensor = Image.fromarray(np.uint8(preprocessed_image)).convert('RGB')
            if debug:
                input_tensor.show()
                input("wait for user input to pass converted to rgb image")
                
            input_tensor = preprocess(input_tensor)
            input_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            if self.use_cuda:
                input_tensor = input_tensor.to('cuda')
                
            output = self.model(input_tensor)
            return output
    
    
    def clear_memory(self):
        print('\n\nmemory before Model deletion:')
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        del self.model
        print('memory after Model deletion:')
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        if self.cam:
            print('\n\nmemory after CAM method deletion:')
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024)) 
            del self.cam
            print('memory before CAM method deletion:')
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        
        gc.collect()
        torch.cuda.empty_cache()
        self.model = None
        self.cam = None
    
        
    def run_menu_no_loop(self, event, call_exit, input_image, offset):
        if not self.main_location:
            self.main_location = offset
        if parameters.call_pause == True:
            last_pause = True
            if not last_pause:
                print('PAUSED')
        else:
            last_pause = False
        
        call_exit = False          
        if event.type == pygame.QUIT:
            call_exit = True
            return call_exit
        elif event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE or event.key == K_q:
                call_exit = True
                return call_exit
            elif event.key == K_SPACE:
                parameters.call_pause = not parameters.call_pause
                if parameters.call_pause:
                    if self.cam is not None:
                        if input_image:
                            class_name, class_score = self.run_cam(input_image)
                            self.last_image_evaluated = input_image
                        else:
                            print('[W] No input image')
                            class_name, class_score = self.run_cam()
                        
                        self.classification_output = f"Class detected: {class_name} with score: {class_score}%"
                        print(self.classification_output)
                    else:
                        no_cam_warning = "CAM method is not selected, Press button M"
                        print(no_cam_warning)
                        draw_text(no_cam_warning, self.font, (255, 255, 255), self.display, 0, 0)
                        pygame.display.update()
                        parameters.call_pause = False
                        
            elif event.key == pygame.K_m:
                if not parameters.call_pause:
                    cam_name = self.select_cam()
                    if cam_name != self.cam_name:
                        self.cam = self.load_cam_if()
                        self.cam_name = cam_name
                        cam_selected = (f'{cam_name} selected, loading...')
                        print(cam_selected)
                        pygame.display.update()
                    else:                        
                        print(f'{cam_name} selected, loaded')
                    return False
                else:
                    print('Comparing with another method')
                    if input_image:
                            class_name, class_score = self.compare_new_method(input_image)
                    else:
                        print('[W] No input image')
                
            elif event.key == pygame.K_n:
                if not parameters.call_pause:
                    self.select_model()
                    return False
            
            elif event.key == pygame.K_t:
                if not parameters.call_pause:
                    print('show tops')
                else:
                    self.get_top_detections(input_image)
                    self.get_detection(input_image)
                return False

            elif event.key == pygame.K_s:
                self.get_detection(input_image)
                return False
                    
                
        if parameters.call_pause:
            self.render(offset)
            pygame.display.update()              
                            
if __name__ == '__main__':
    pygame.init()
    pygame.font.init() #for fonts rendering
    display = pygame.display.set_mode([1920,1080], pygame.HWSURFACE | pygame.DOUBLEBUF)
    test_menu = menu(display)
    call_exit = False
    sample_image = pygame.image.load('/home/roc/tfm/XAI-Visualizer/input_images/carla_input/1.png')
    display.blit(sample_image, [0,0])
    pygame.display.update()
    input("enter to pass loaded image")
    while not call_exit:
        for event in pygame.event.get():    
            call_exit = test_menu.run_menu_no_loop(event, call_exit, sample_image, [0,0])
