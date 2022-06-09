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
from torchvision import models
import parameters

from parameters import BUTTON_COLOR, WHITE

import pickle
from urllib.request import urlopen

import subprocess
from time import sleep

class menu:
    def __init__(self, display, font, model = models.resnet18(pretrained=True)):
        self.model = model
        self.display = display
        self.font = font
        self.target_layers = [self.model.layer4[-1]]
        self.cam_name = "0"
        self.CAM_BUTTON_COLOR = (169,169,169)
        self.MODEL_BUTTON_COLOR = (169,169,169)
        self.click = False
        
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
            
            self.display.fill((0,0,0))
            draw_text('Method Menu', self.font, (255, 255, 255), self.display, 20, 20)
            
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
                    print(f'{method_name} selected, loading...')
                    
            if score_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'ScoreCAM'
                    print(f'{method_name} selected, loading...')
                    
            if ablation_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'AblationCAM'
                    print(f'{method_name} selected, loading...')
                    
            if xgradcam_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'XGradCAM'
                    print(f'{method_name} selected, loading...')
                    
            if eigen_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'EigenCAM'
                    print(f'{method_name} selected, loading...')
                    
            if fullgrad_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'FullGrad'
                    print(f'{method_name} selected, loading...')
                    
            if gradcampp_button.collidepoint((mx, my)):
                if self.click:
                    method_selection = False
                    method_name = 'GradCAM++'
                    print(f'{method_name} selected, loading...')
                    
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
            
        return cam_method
        
    def select_target_layer(self):
        self.target_layers = [self.model.layer4[-1]]
    
    def select_model(self):
        self.click = False
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
            
            self.display.fill((0,0,0))
            draw_text('model Menu', self.font, (255, 255, 255), self.display, 20, 20)
            
            mx, my = pygame.mouse.get_pos()
            # To delimit the size of the button, in the future use value related to window res
            w, h = pygame.display.get_surface().get_size()
            button_width = 300
            button_height = 40
            
            resnet_button = pygame.Rect(positions[0][0], positions[0][1], button_width, button_height)
            score_button = pygame.Rect(positions[1][0], positions[1][1], button_width, button_height)
            xgradcam_button = pygame.Rect(positions[2][0], positions[2][1], button_width, button_height)
            ablation_button = pygame.Rect(positions[3][0], positions[3][1], button_width, button_height)

            pygame.draw.rect(self.display, BUTTON_COLOR,  resnet_button)
            draw_text('GradCAM', self.font, (255, 255, 255), self.display, positions[0][0], positions[0][1]+button_height-15)
            pygame.draw.rect(self.display, BUTTON_COLOR, score_button)
            draw_text('ScoreCAM', self.font, (255, 255, 255), self.display, positions[1][0], positions[1][1]+button_height-15)
            pygame.draw.rect(self.display, BUTTON_COLOR, xgradcam_button)
            draw_text('XGradCAM', self.font, (255, 255, 255), self.display, positions[2][0], positions[2][1]+button_height-15)
            pygame.draw.rect(self.display, BUTTON_COLOR, ablation_button)
            draw_text('AblationCAM', self.font, (255, 255, 255), self.display, positions[3][0], positions[3][1]+button_height-15)

            if  resnet_button.collidepoint((mx, my)):
                if self.click:
                    model = models.resnet(pretrained=True)
                    model_selection = False
                    model_name = 'ResNet'
                    print(f'{model_name} selected, loading...')
                    
            if score_button.collidepoint((mx, my)):
                if self.click:
                    model = models.alexnet(pretrained=True) 
                    model_selection = False
                    model_name = 'Alexnet'
                    print(f'{model_name} selected, loading...')
                    
            if ablation_button.collidepoint((mx, my)):
                if self.click:
                    model = models.alexnet(pretrained=True) 
                    model_selection = False
                    model_name = 'Alexnet'
                    print(f'{model_name} selected, loading...')
                    offsetpos = 4
            if xgradcam_button.collidepoint((mx, my)):
                if self.click:
                    model = models.alexnet(pretrained=True) 
                    model_selection = False
                    model_name = 'Alexnet'
                    print(f'{model_name} selected, loading...')
                    
                    
                    
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

            return model
        
    def run_menu(self):
        call_exit = False
        while not call_exit:            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break
                    elif event.key == K_SPACE:
                        parameters.call_pause = not parameters.call_pause
                    elif event.key == pygame.K_m:
                        cam_name = self.select_cam(self)
                        if cam_name != self.cam_name:
                            self.cam = menu.load_cam(self)
                            self.cam_name = cam_name
                            print(f'{cam_name} selected, loading...')
                        else:                        
                            print(f'{cam_name} selected, loaded')
                    elif event.key == pygame.K_n:
                        model = self.select_model(self)
                        if model.name != self.model.name:
                            self.model = model
                            print(f'{model.name} selected, loading...')
                        else:                        
                            print(f'{model.name} selected, loaded')
                        
