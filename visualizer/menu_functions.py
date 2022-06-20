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

def_img = "../input_images/carla_input/1.png"

class menu:
    def __init__(self, display, world = None):
        self.model = resnet50(pretrained=True)
        self.display = display
        self.world = world
        self.font = pygame.font.SysFont(None, 24)
        self.target_layers = self.select_target_layer()
        self.cam_name = "0"
        self.CAM_BUTTON_COLOR = BUTTON_COLOR
        self.MODEL_BUTTON_COLOR = BUTTON_COLOR
        self.click = False
        self.class_list = roc_functions.get_imagenet_dictionary(url=None) 
        self.cam = None
        if torch.cuda.is_available():
            self.use_cuda = True
            self.model.to('cuda')
            print("System is cuda ready")
        else:
            self.use_cuda = False
        self.surface = pygame.image.load(def_img)
        # cool graphic:
        # x = np.arange(0, 1920)
        # y = np.arange(0, 1080)
        # X, Y = np.meshgrid(x, y)
        # Z = X + Y
        # Z = 255*Z/Z.max()
        # self.surface = pygame.surfarray.make_surface(Z)
        self.image_location = (0,0)
        
        
    def select_cam(self):
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
            draw_text('GradCAM++', self.font, (255, 255, 255), self.display, positions[5][0], positions[5][1]+button_height-15)

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

        return method_name
    
    def load_cam(self, method_name):
        
        if self.use_cuda:
            print('Memory Summary before loading CAM:')
            print(torch.cuda.memory_summary(device='cuda', abbreviated=False))    
        if method_name == 'ScoreCAM':
            cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            
        if method_name == 'AblationCAM':
            cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            
        if method_name == 'XGradCAM':
            cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            
        if method_name == 'EigenCAM':
            cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            
        if method_name == 'FullGrad':
            cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=True)
            
        if method_name == 'GradCAM++':
            cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=True)
            
        if method_name == 'GradCAM':
            cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
        
        if self.use_cuda:
            print('Memory Summary after loading CAM:')
            print(torch.cuda.memory_summary(device='cuda', abbreviated=False))   
            
        return cam_method
        
    def select_target_layer(self):
        # to implement for multiple models
        if self.model.__class__.__name__ == 'ResNet':
            return [self.model.layer4[-1]]
        elif self.model.__class__.__name__ == 'Alexnet':
            return [11]
    
    
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
            draw_text('Alexnet', self.font, (255, 255, 255), self.display, positions[2][0], positions[2][1]+button_height-15)
            pygame.draw.rect(self.display, BUTTON_COLOR, fourth_button)
            draw_text('Alexnet', self.font, (255, 255, 255), self.display, positions[3][0], positions[3][1]+button_height-15)

            if resnet_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'ResNet':
                        self.model = resnet34(pretrained=True)
                        model_selection = False
                        model_name = 'ResNet'
                        if self.use_cuda:
                            print('Memory Summary before loading model:')
                            print(torch.cuda.memory_summary(device='cuda', abbreviated=False)) 
                            self.model.to('cuda')
                            print(f"Selected model {model_name} is cuda ready")
                            print('Memory Summary after loading model:')
                            print(torch.cuda.memory_summary(device='cuda', abbreviated=False)) 
                    else:
                        print(f'Model selected -> {self.model.__class__.__name__} was already loaded')
            
            if alexnet_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'Alexnet':
                        self.model = alexnet(pretrained=True) 
                        model_selection = False
                        model_name = 'Alexnet'
                        if self.use_cuda:
                            print('Memory Summary before loading model:')
                            print(torch.cuda.memory_summary(device='cuda', abbreviated=False)) 
                            self.model.to('cuda')
                            print(f"Selected model {model_name} is cuda ready")
                            print('Memory Summary after loading model:')
                            print(torch.cuda.memory_summary(device='cuda', abbreviated=False)) 
                    else:
                        print(f'Model selected -> {self.model.__class__.__name__} was already loaded')
            
            if third_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'Alexnet':
                        self.model = alexnet(pretrained=True) 
                        model_selection = False
                        model_name = 'Alexnet'
                        if self.use_cuda:
                            print('Memory Summary before loading model:')
                            print(torch.cuda.memory_summary(device='cuda', abbreviated=False)) 
                            self.model.to('cuda')
                            print(f"Selected model {model_name} is cuda ready")
                            print('Memory Summary after loading model:')
                            print(torch.cuda.memory_summary(device='cuda', abbreviated=False)) 
                    else:
                        print(f'Model selected -> {self.model.__class__.__name__} was already loaded')
                    
            if fourth_button.collidepoint((mx, my)):
                if self.click:
                    if self.model.__class__.__name__ != 'Alexnet':
                        self.model = alexnet(pretrained=True) 
                        model_selection = False
                        model_name = 'Alexnet'
                        if self.use_cuda:
                            print('Memory Summary before loading model:')
                            print(torch.cuda.memory_summary(device='cuda', abbreviated=False)) 
                            self.model.to('cuda')
                            print(f"Selected model {model_name} is cuda ready")
                            print('Memory Summary after loading model:')
                            print(torch.cuda.memory_summary(device='cuda', abbreviated=False)) 
                    else:
                        print(f'Model selected -> {self.model.__class__.__name__} was already loaded')
            
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

        # if model:
        #     return model
    
    def prob_calc(self, img = pygame.image.load(def_img)):
        probabilities = self.run_model(img)
        probabilities = probabilities.to('cpu')
        target_class = np.argmax(probabilities.data.numpy())
        class_name = self.class_list[target_class]
        class_score = probabilities[target_class]
        return class_name, class_score
    
    def run_cam(self, img = pygame.image.load(def_img)):
        
        t0 = time.time()
        # get the cam heat map in a pygame image
        self.surface =  roc_functions.surface_to_cam(img, self.cam)
        print('time needed for visualization method creation :', time.time()-t0)
        t1 = time.time()
        class_name, class_score = self.prob_calc(img)
        print('time needed for probabilities calculation:', time.time()-t1)
        self.image_location = (0,0)
        self.render
        pygame.display.update()
        return class_name, class_score
        
    def run_cam_parallel(self, img = pygame.image.load(def_img)):
        
        # get the cam heat map in a pygame image
        process_cam = Process(target=roc_functions.surface_to_cam, args=(img, self.cam))
        process_prob = Process(target=self.prob_calc, args=(self.model, img))
        t0 = time.time()
        process_cam.start()
        process_prob.start()
        
        process_cam.join()
        print('time needed for visualization method creation :', time.time()-t0)
        
        process_prob.join()
        print('time needed for probabilities calculation:', time.time()-t0)
        
        
    def render(self):
        if self.surface is not None:
            self.display.blit(self.surface, self.image_location)

        
    def run_model(self, img = pygame.image.load(def_img)):
        with torch.no_grad():
            preprocessed_image = pygame.surfarray.pixels3d(img)
            preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            input_tensor = Image.fromarray(np.uint8(preprocessed_image)).convert('RGB')
            del preprocessed_image
            input_tensor = preprocess(input_tensor)
            input_tensor = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            if self.use_cuda:
                input_tensor = input_tensor.to('cuda')
            output = self.model(input_tensor)
            del input_tensor
            # The output has unnormalized scores. To get probabilities, run a softmax on it.
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            return probabilities
        
    # def run_menu(self):
    #     call_exit = False
    #     while not call_exit:            
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 call_exit = True
    #             elif event.type == pygame.KEYDOWN:
    #                 if event.key == K_ESCAPE or event.key == K_q:
    #                     call_exit = True
    #                     break
    #                 elif event.key == K_SPACE:
    #                     parameters.call_pause = not parameters.call_pause
    #                     if parameters.call_pause:
    #                         if self.cam is not None:
    #                             class_name, class_score = self.run_cam()
    #                             print(f"Class detected: {class_name}")
    #                         else:
    #                             print("CAM method is not selected, Press button M")
    #                             parameters.call_pause = False
                                
    #                 elif event.key == pygame.K_m:
    #                     if not parameters.call_pause:
    #                         cam_name = self.select_cam()
    #                         if cam_name != self.cam_name:
    #                             self.cam = self.load_cam(cam_name)
    #                             self.cam_name = cam_name
    #                             print(f'{cam_name} selected, loading...')
    #                         else:                        
    #                             print(f'{cam_name} selected, loaded')
    #                         break
                        
    #                 elif event.key == pygame.K_n:
    #                     if not parameters.call_pause:
    #                         model = self.select_model()
    #                         if model.__class__.__name__ != self.model.__class__.__name__:
    #                             self.model = model
    #                             print(f'{model.__class__.__name__} selected, loading...')
    #                         else:                        
    #                             print(f'{model.__class__.__name__} selected, loaded')
    #                         break
    #         self.render()
    #         pygame.display.update()
            
    def run_menu_no_loop(self, event, call_exit):            
        if event.type == pygame.QUIT:
            call_exit = True
        elif event.type == pygame.KEYDOWN:
            if event.key == K_ESCAPE or event.key == K_q:
                call_exit = True
                return call_exit
            elif event.key == K_SPACE:
                parameters.call_pause = not parameters.call_pause
                if parameters.call_pause:
                    if self.cam is not None:
                        class_name, class_score = self.run_cam()
                        print(f"Class detected: {class_name}")
                    else:
                        print("CAM method is not selected, Press button M")
                        parameters.call_pause = False
                        
            elif event.key == pygame.K_m:
                if not parameters.call_pause:
                    cam_name = self.select_cam()
                    if cam_name != self.cam_name:
                        self.cam = self.load_cam(cam_name)
                        self.cam_name = cam_name
                        print(f'{cam_name} selected, loading...')
                    else:                        
                        print(f'{cam_name} selected, loaded')
                    return call_exit
                
            elif event.key == pygame.K_n:
                if not parameters.call_pause:
                    self.select_model()
                    return call_exit
                
        self.render()
        pygame.display.update()           
                            
if __name__ == '__main__':
    pygame.init()
    pygame.font.init() #for fonts rendering
    display = pygame.display.set_mode([1920,1080], pygame.HWSURFACE | pygame.DOUBLEBUF)
    test_menu = menu(display)
    call_exit = False
    while not call_exit:
        for event in pygame.event.get():    
            call_exit = test_menu.run_menu_no_loop(event, call_exit)
