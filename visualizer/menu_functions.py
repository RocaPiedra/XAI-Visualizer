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

class menu:
    def __init__(self, display, world = None, use_cuda = True):
        self.model = resnet18(pretrained=True)
        self.display = display
        self.world = world
        self.use_cuda = use_cuda
        self.font = pygame.font.SysFont(None, 24)
        self.target_layers = self.select_target_layer()
        self.cam_name = "0"
        self.CAM_BUTTON_COLOR = BUTTON_COLOR
        self.MODEL_BUTTON_COLOR = BUTTON_COLOR
        self.click = False
        self.class_list = roc_functions.get_imagenet_dictionary(url=None) 
        self.cam = None
        if torch.cuda.is_available() and self.use_cuda:
            self.model.to('cuda')
            print("System is cuda ready")
            
        self.surface = None
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
            draw_text('GradCAM++', self.font, (255, 255, 255), self.display, positions[6][0], positions[6][1]+button_height-15)

            if grad_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    self.method_name = 'GradCAM'
                    
            if score_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    self.method_name = 'ScoreCAM'
                    
            if ablation_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    self.method_name = 'AblationCAM'
                    
            if xgradcam_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    self.method_name = 'XGradCAM'
                    
            if eigen_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    self.method_name = 'EigenCAM'
                    
            if fullgrad_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    self.method_name = 'FullGrad'
                    
            if gradcampp_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    self.method_name = 'GradCAM++'
                    
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

        return self.method_name
    
    def load_cam(self):
        # if self.use_cuda:
        #     print('Memory Summary before loading CAM:')
        #     print(torch.cuda.memory_summary(device='cuda', abbreviated=True))    
        if self.method_name == 'ScoreCAM':
            try:
                cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=self.use_cuda)
            except:
                print("error thrown, using CPU")
                cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'AblationCAM':
            try:
                cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=self.use_cuda)
            except:
                print("error thrown, using CPU")
                cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'XGradCAM':
            try:
                cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=self.use_cuda)
            except:
                print("error thrown, using CPU")
                cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'EigenCAM':
            try:
                cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=self.use_cuda)
            except:
                print("error thrown, using CPU")
                cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'FullGrad':
            try:
                cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=self.use_cuda)
            except:
                print("error thrown, using CPU")
                cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'GradCAM++':
            try:
                cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=self.use_cuda)
            except:
                print("error thrown, using CPU")
                cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'GradCAM':
            try:
                cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=self.use_cuda)
            except:
                print("error thrown, using CPU")
                cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        # if self.use_cuda:
        #     print('Memory Summary after loading CAM:')
        #     print(torch.cuda.memory_summary(device='cuda', abbreviated=False))   
            
        return cam_method
    
    def load_cam_if(self, cuda_error = False):
        
        # if self.use_cuda:
        #     print('Memory Summary before loading CAM:')
        #     print(torch.cuda.memory_summary(device='cuda', abbreviated=True))
            
            
        if self.method_name == 'ScoreCAM':
            if not cuda_error:
                cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = ScoreCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'AblationCAM':
            if not cuda_error:
                cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = AblationCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'XGradCAM':
            if not cuda_error:
                cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = XGradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'EigenCAM':
            if not cuda_error:
                cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = EigenCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'FullGrad':
            if not cuda_error:
                cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = FullGrad(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'GradCAM++':
            if not cuda_error:
                cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = GradCAMPlusPlus(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        if self.method_name == 'GradCAM':
            if not cuda_error:
                cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=True)
            else:
                print("error thrown, using CPU")
                cam_method = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=False)
            
        # if self.use_cuda and not cuda_error:
        #     print('Memory Summary after loading CAM:')
        #     print(torch.cuda.memory_summary(device='cuda', abbreviated=False))   
            
        return cam_method
        
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
                        try:
                            print('Reloading CAM method with new Model')
                            self.cam = self.load_cam_if()
                        except:
                            print('Some error ocurred, try loading cam to CPU')
                            self.cam = self.load_cam_if(True)
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
                        try:
                            print('Reloading CAM method with new Model')
                            self.cam = self.load_cam_if()
                        except:
                            print('Some error ocurred, try loading cam to CPU')
                            self.cam = self.load_cam_if(True)
                            
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
                        try:
                            print('Reloading CAM method with new Model')
                            self.cam = self.load_cam_if()
                        except:
                            print('Some error ocurred, try loading cam to CPU')
                            self.cam = self.load_cam_if(True)
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
                        try:
                            print('Reloading CAM method with new Model')
                            self.cam = self.load_cam_if()
                        except:
                            print('Some error ocurred, try loading cam to CPU')
                            self.cam = self.load_cam_if(True)
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
            
        return model_name

    #This option obtains the inference results from outside the cam method
    def prob_calc(self, img):
        output = self.run_model(img)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probabilities = probabilities.to('cpu')
        target_class = np.argmax(probabilities.data.numpy())
        class_name = self.class_list[target_class]
        class_score = probabilities[target_class]
        return class_name, class_score
    
    def prob_calc_efficient(self, output):
        # The output has unnormalized scores. To get probabilities, run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        probabilities = probabilities.to('cpu')
        target_class = np.argmax(probabilities.data.numpy())
        class_name = self.class_list[target_class]
        class_score = probabilities[target_class]
        return class_name, class_score.cpu().detach().numpy()
    
    def run_cam(self, img):
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.time()
        # get the cam heat map in a pygame image
        self.surface, inf_outputs =  roc_functions.surface_to_cam(img, self.cam, self.use_cuda)
        print('time needed for visualization method creation :', time.time()-t0)
        t1 = time.time()
        class_name, class_score = self.prob_calc_efficient(inf_outputs)
        class_percentage = str(round(class_score*100,2))
        print('time needed for probabilities calculation:', time.time()-t1)
        self.image_location = (0,0)
        self.render
        pygame.display.update()
        return class_name, class_percentage

    def render(self, selected_location):
        if selected_location:
            self.image_location = selected_location
        if self.surface is not None:
            self.display.blit(self.surface, self.image_location)
    
    def run_model(self, img):
        with torch.no_grad():
            preprocessed_image = pygame.surfarray.pixels3d(img)
            preprocess = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])
            input_tensor = Image.fromarray(np.uint8(preprocessed_image)).convert('RGB')
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
        classification_output = ''  
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
                        else:
                            print('[W] No input image')
                            class_name, class_score = self.run_cam()
                        
                        classification_output = f"Class detected: {class_name} with score: {class_score}%"
                        print(classification_output)
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
                        self.cam = self.load_cam()
                        self.cam_name = cam_name
                        cam_selected = (f'{cam_name} selected, loading...')
                        print(cam_selected)
                        pygame.display.update()
                    else:                        
                        print(f'{cam_name} selected, loaded')
                    return False
                
            elif event.key == pygame.K_n:
                if not parameters.call_pause:
                    self.select_model()
                    return False
                
        if parameters.call_pause:
            # draw_text(classification_output, self.font, (255, 255, 255), self.display, offset[0], offset[1])
            self.render(offset)
            font_output = self.font.render(classification_output, True, (255, 255, 255))
            self.display.blit(font_output, offset)
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
