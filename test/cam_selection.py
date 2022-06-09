import numpy as np
import sys

sys.path.append('../visualizer')

import roc_functions
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
from torchvision import transforms

# import torch
import numpy as np
import time

try:
    import pygame
    import pygame.camera
    import pygame.image
    from pygame.locals import *
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

WHITE = (255, 255, 255)

def get_offset_list(window_res, image_res):
    grid_size = [int(np.fix(window_res[0]/image_res[0])), int(np.fix(window_res[1]/image_res[1]))]
    offset_list = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            offset = (i*image_res[0], j*image_res[1])
            offset_list.append(tuple(offset))
    window_size = [grid_size[0]*image_res[0], grid_size[1]*image_res[1]]
    return offset_list, window_size

def CAM_pygame():
    pygame.init()
    pygame.font.init() #for fonts rendering
    pygame.camera.init()

    font = pygame.font.SysFont(None, 24)
    
    cameras = pygame.camera.list_cameras()
    res = '1920x1080'
    window_size = [int(x) for x in res.split('x')]
    webcam = pygame.camera.Camera(cameras[0])
    pause = False
    imagenet = roc_functions.get_imagenet_dictionary(url=None)                
    model = resnet18(pretrained=True)
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    target_layers = [model.layer4[-1]]
    
    targets = ClassifierOutputTarget(281)

    webcam.start()
    img = webcam.get_image()
    image_size = [img.get_width(), img.get_height()]
    offset_list, window_size = get_offset_list(window_size, image_size)
    display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("PyGame Camera View")
    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_m:
                    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)

                if event.key == K_SPACE:
                    pause = True                    
                    with torch.no_grad():
                        preprocessed_image = pygame.surfarray.pixels3d(img)
                        preprocess = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
                        PIL_image = Image.fromarray(np.uint8(preprocessed_image)).convert('RGB')
                        del preprocessed_image
                        input_tensor = preprocess(PIL_image)
                        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

                        if torch.cuda.is_available():
                            # tensor = tensor.to('cuda')
                            input_batch = input_batch.to('cuda')
                        output = model(input_batch)

                    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    probabilities = probabilities.to('cpu')
                    target_class = np.argmax(probabilities.data.numpy())
                    class_name = imagenet[target_class]
                    class_score = probabilities[target_class]  
                    t0 = time.time()
                    cam_surface = roc_functions.surface_to_cam(img, cam)
                    print('time needed for fullgrad creation :', time.time()-t0)
                    display.blit(cam_surface,offset_list[pos])
                    full_name = name + ': '+class_name+'| Score: '+str(round(float(class_score)*100,2))+'%'
                    print(full_name)
                    text = font.render(full_name, True, WHITE)
                    display.blit(text,offset_list[pos])
                    pygame.display.flip()
                    
                    while pause:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                sys.exit()
                            if event.type == pygame.KEYDOWN:
                                if event.key == K_SPACE:
                                    pause = False
                                if event.key == K_ESCAPE:
                                    pygame.quit()
                                    sys.exit()
                                if event.key == K_m:
                                    # pause = False
                                    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)

        for offset in offset_list:
            display.blit(img,offset)
        pygame.display.flip()
        img = webcam.get_image()

def CAM_pygame_loop(font, webcam, model, display, target_layers, imagenet, window_size):

    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    img = webcam.get_image()
    image_size = [img.get_width(), img.get_height()]
    offset_list, window_size = get_offset_list(window_size, image_size)
    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_m:
                    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)

                if event.key == K_SPACE:
                    pause = True                    
                    with torch.no_grad():
                        preprocessed_image = pygame.surfarray.pixels3d(img)
                        preprocess = transforms.Compose([
                                    transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
                        PIL_image = Image.fromarray(np.uint8(preprocessed_image)).convert('RGB')
                        del preprocessed_image
                        input_tensor = preprocess(PIL_image)
                        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

                        if torch.cuda.is_available():
                            # tensor = tensor.to('cuda')
                            input_batch = input_batch.to('cuda')
                        output = model(input_batch)

                    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    probabilities = probabilities.to('cpu')
                    target_class = np.argmax(probabilities.data.numpy())
                    class_name = imagenet[target_class]
                    class_score = probabilities[target_class]  
                    t0 = time.time()
                    cam_surface = roc_functions.surface_to_cam(img, cam)
                    print('time needed for fullgrad creation :', time.time()-t0)
                    display.blit(cam_surface,offset_list[pos])
                    full_name = name + ': '+class_name+'| Score: '+str(round(float(class_score)*100,2))+'%'
                    print(full_name)
                    text = font.render(full_name, True, WHITE)
                    display.blit(text,offset_list[pos])
                    pygame.display.flip()
                    
                    while pause:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                sys.exit()
                            if event.type == pygame.KEYDOWN:
                                if event.key == K_SPACE:
                                    pause = False
                                if event.key == K_ESCAPE:
                                    pygame.quit()
                                    sys.exit()
                                if event.key == K_m:
                                    # pause = False
                                    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)

        for offset in offset_list:
            display.blit(img,offset)
        pygame.display.flip()
        img = webcam.get_image()
    

if __name__ == '__main__':
    pygame.init()
    pygame.font.init() #for fonts rendering
    pygame.camera.init()

    font = pygame.font.SysFont(None, 24)
    
    cameras = pygame.camera.list_cameras()
    res = '1920x1080'
    window_size = [int(x) for x in res.split('x')]

    webcam = pygame.camera.Camera(cameras[0])
    pause = False
    imagenet = roc_functions.get_imagenet_dictionary(url=None)                
    model = resnet18(pretrained=True)

    display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
    
    pygame.display.set_caption("Visualizer")

    target_layers = [model.layer4[-1]]
    
    targets = ClassifierOutputTarget(281)

    webcam.start()

    img = webcam.get_image()
    image_size = [img.get_width(), img.get_height()]
    offset_list, window_size = get_offset_list(window_size, image_size)

    CAM_pygame_loop(font, webcam, model, display, target_layers, imagenet, window_size)