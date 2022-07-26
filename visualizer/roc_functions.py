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

import torch
from torch.autograd import Variable
from torchvision import models
import parameters

import pickle
from urllib.request import urlopen

import subprocess, signal
from time import sleep



def preprocess_image(pil_im, sendToGPU=True, resize_im=True):
    """
        Processes image for CNNs

    Args:
        PIL_img (PIL_img): PIL Image or numpy array to process
        sendToGPU (bool): To process in GPU or not
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (torch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    #ensure or transform incoming image to PIL image
    if type(pil_im) != Image.Image:
        try:
            pil_im = Image.fromarray(pil_im)
        except Exception as e:
            print("could not transform PIL_img to a PIL Image object. Please check input.")

    # Resize image
    if resize_im:
        pil_im = pil_im.resize((224, 224), Image.ANTIALIAS)

    im_as_arr = np.float32(pil_im) # W,H,D
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    # if not im_as_var.is_cuda() and not sendtoGPU:
    if sendToGPU:
        im_as_var.to('cuda')
    return im_as_var

def get_image_path(path, filename):
    if filename == None:
        onlyimages = [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) & f.endswith(('.jpg','.png'))]
        return onlyimages
    else:
        image_path = path + filename
        return image_path

def choose_model(input_argument = None, model_name = None):
    if model_name == 'resnet':
        return models.resnet50(pretrained=True)
    elif model_name == 'alexnet':
        return models.alexnet(pretrained=True)
    elif model_name == 'yolov5':
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    elif input_argument is not None:
        if int(input_argument) in (1,2,3):
            option = int(input_argument)
    else:
        option = int(input('Model is not defined choose from the available list:\n'
    '1. Alexnet\n2. ResNet\n3. YoloV5\n'))
    if option == 1:
        print('Alexnet is the chosen classifier')
        return models.alexnet(pretrained=True)
    elif option == 2:
        print('ResNet is the chosen classifier')
        return models.resnet18(pretrained=True)
    elif option == 3:
        print('YoloV5 is the chosen classifier')
        return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    else:
        print('Option incorrect, set default model: Alexnet')
        return models.alexnet(pretrained=True)

def get_top_classes(output, number_of_classes = 5):
    idx = np.argpartition(output, -number_of_classes)[-number_of_classes:]
    return idx

def get_class_name_imagenet(idx):
    try:
        imagenet = pickle.load('../ImageNet_utils/imagenet1000_clsid_to_human.pkl')

    except:
        url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'
        imagenet = pickle.load(urlopen(url))

    return imagenet[idx]

def get_imagenet_dictionary(url=None):
    if url is None:
        url = 'https://gist.githubusercontent.com/yrevar/6135f1bd8dcf2e0cc683/raw/d133d61a09d7e5a3b36b8c111a8dd5c4b5d560ee/imagenet1000_clsid_to_human.pkl'
    try:
        path_name = os.path.realpath(os.path.join(os.path.dirname(__file__),'../ImageNet_utils/imagenet1000_clsid_to_human.pkl'))
        # print(f'path is: {path_name}\nexists? -> {os.path.isfile(path_name)}')
        imagenet = pickle.load(path_name)
    except:
        print('pickle file was not found, downloading...')
        imagenet = pickle.load(urlopen(url))

    return imagenet

def launch_carla_simulator_locally(unreal_engine_path = parameters.unreal_engine_path):
    sim_running = False
    print('Launching Unreal Engine Server...')
    if os.name == 'nt':
        unreal_engine = subprocess.Popen(unreal_engine_path, stdout=subprocess.PIPE)
    else:
        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            if 'CarlaUE4-Linux-' in str(line):
                print('Simulator is already running')
                sim_running = True
                p.terminate()
        if not sim_running:
            unreal_engine = subprocess.Popen([unreal_engine_path], stdout=subprocess.PIPE)
    sleep(5)
    print('Generating traffic...')
    generate_traffic = subprocess.Popen(["python", "../carlacomms/generate_traffic.py", '--asynch', '--tm-port=8001'], stdout=subprocess.PIPE)
    return unreal_engine, generate_traffic

def close_carla_simulator():
    if os.name == 'nt':
        print('windows termination process:')
        os.system('taskkill /f /im CarlaUE4.exe')
    else:
        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        print('linux termination process:')
        for line in out.splitlines():
            if 'CarlaUE4-Linux-' in str(line):
                pid = int(line.split(None, 1)[0])
                os.kill(pid, signal.SIGKILL)

def get_offset_list(window_res, image_res):
    grid_size = [int(np.fix(window_res[0]/image_res[0])), int(np.fix(window_res[1]/image_res[1]))]
    offset_list = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            offset = (i*image_res[0], j*image_res[1])
            offset_list.append(tuple(offset))
    window_size = [grid_size[0]*image_res[0], grid_size[1]*image_res[1]]
    return offset_list, window_size

def surface_to_cam(surface, cam_method, use_cuda=True):
    array = pygame.surfarray.pixels3d(surface)
    normalized_image = np.float32(array/255)
    input_tensor = preprocess_image(array, use_cuda, False)
    
    print(f'Verify input tensor and model location, GPU usage selected: {use_cuda}')
    print(f'Input Tensor is in GPU: {input_tensor.is_cuda}')
    print(f'Model is in GPU: {next(cam_method.model.parameters()).is_cuda}')
    
    if input_tensor.is_cuda != next(cam_method.model.parameters()).is_cuda:
        print('The input and the model location do not match. Trying to solve the problem...')
        if use_cuda:
            input_tensor.to('cuda')
            cam_method.model.to('cuda')
            # cam_method.model.cuda() # Does the same
            print('Input and model moved to GPU')
        else:
            input_tensor.to('cpu')
            cam_method.model.to('cpu')
            print('Input and model moved to CPU')
    
        print(f'New location:')
        print(f'Input Tensor is in GPU: {input_tensor.is_cuda}')
        print(f'Model is in GPU: {next(cam_method.model.parameters()).is_cuda}')
    
    try:
        grayscale_cam, inf_outputs = cam_method(input_tensor)
        
    except KeyboardInterrupt:
        print('Closing app')
        
    except:
        print(f'Exception handled for input tensor not matching location,\
            is it cuda? -> {input_tensor.is_cuda}, change location')
        if input_tensor.is_cuda:
            input_tensor.to('cpu')
        else:
            try:
                input_tensor.to('cuda')
            except:
                print('no memory available for GPU')
                input_tensor.to('cpu') #could mismatch tensor and cam method which cannot be sent to cpu from here
        grayscale_cam, inf_outputs = cam_method(input_tensor)
        
    print(f'CAM Generated for model {cam_method.model.__class__.__name__}')
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(normalized_image, grayscale_cam, use_rgb=True)
    cam_surface = pygame.surfarray.make_surface(visualization)
    return cam_surface, inf_outputs

def draw_text(text, font, color, surface, x, y):
    textobj = font.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

# GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
WHITE = (255, 255, 255)
BUTTON_COLOR = (169,169,169)

def method_menu(font, surface, model, target_layers):
    click = False
    method_selection = True
    positions = [[100,100],
                [100,200],
                [100,300],
                [100,400],
                [100,500],
                [100,600]]

    while method_selection:
        
        surface.fill((0,0,0))
        draw_text('Method Menu', font, (255, 255, 255), surface, 20, 20)
        
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

        pygame.draw.rect(surface, BUTTON_COLOR, grad_button)
        draw_text('GradCAM', font, (255, 255, 255), surface, positions[0][0], positions[0][1]+button_height-15)
        pygame.draw.rect(surface, BUTTON_COLOR, score_button)
        draw_text('ScoreCAM', font, (255, 255, 255), surface, positions[1][0], positions[1][1]+button_height-15)
        pygame.draw.rect(surface, BUTTON_COLOR, xgradcam_button)
        draw_text('XGradCAM', font, (255, 255, 255), surface, positions[2][0], positions[2][1]+button_height-15)
        pygame.draw.rect(surface, BUTTON_COLOR, ablation_button)
        draw_text('AblationCAM', font, (255, 255, 255), surface, positions[3][0], positions[3][1]+button_height-15)
        pygame.draw.rect(surface, BUTTON_COLOR, eigen_button)
        draw_text('EigenCAM', font, (255, 255, 255), surface, positions[4][0], positions[4][1]+button_height-15)
        pygame.draw.rect(surface, BUTTON_COLOR, fullgrad_button)
        draw_text('FullGrad', font, (255, 255, 255), surface, positions[5][0], positions[5][1]+button_height-15)

        if grad_button.collidepoint((mx, my)):
            if click:
                cam_method = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
                method_selection = False
                method_name = 'GradCAM'
                print(f'{method_name} selected, loading...')
                offsetpos = 0
        if score_button.collidepoint((mx, my)):
            if click:
                cam_method = ScoreCAM(model=model, target_layers=target_layers, use_cuda=True)
                method_selection = False
                method_name = 'ScoreCAM'
                print(f'{method_name} selected, loading...')
                offsetpos = 1
        if ablation_button.collidepoint((mx, my)):
            if click:
                cam_method = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)
                method_selection = False
                method_name = 'AblationCAM'
                print(f'{method_name} selected, loading...')
                offsetpos = 4
        if xgradcam_button.collidepoint((mx, my)):
            if click:
                cam_method = XGradCAM(model=model, target_layers=target_layers, use_cuda=True)
                method_selection = False
                method_name = 'XGradCAM'
                print(f'{method_name} selected, loading...')
                offsetpos = 2
        if eigen_button.collidepoint((mx, my)):
            if click:
                cam_method = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
                method_selection = False
                method_name = 'EigenCAM'
                print(f'{method_name} selected, loading...')
                offsetpos = 3
        if fullgrad_button.collidepoint((mx, my)):
            if click:
                cam_method = FullGrad(model=model, target_layers=target_layers, use_cuda=True)
                method_selection = False
                method_name = 'FullGrad'
                print(f'{method_name} selected, loading...')
                offsetpos = 5
        click = False
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
                    click = True
        pygame.display.update()

    return cam_method, method_name, offsetpos

def check_pytorch_cuda_memory():
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB\n\n"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
    print(f'mem get info output: {torch.cuda.mem_get_info(0)}')