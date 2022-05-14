from concurrent.futures import process
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

def get_probabilities(model, img):
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

        if torch.cuda.is_available():
            input_tensor = input_tensor.to('cuda')
        output = model(input_tensor)
        del input_tensor
        # The output has unnormalized scores. To get probabilities, run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return probabilities

def check_keyboard(event, cam_surface, surface_list, pos_list, pos, pause, process_same_image, display_new_image):   
    print('checking keyboard') 
    
    if event.type == pygame.KEYDOWN :
        if event.key == K_SPACE:
            print("Space pressed")
            pause = False
            process_same_image = False
        if event.key == K_ESCAPE:
            pygame.quit()
            sys.exit()
        if event.key == K_m:
            print("M pressed")
            process_same_image = False
            surface_list = []
            pos_list = []
            cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)
        if event.key == K_n:
            print("N pressed")
            surface_list.append(cam_surface)
            pos_list.append(offset_list[pos])
            process_same_image = True
            display_new_image = True
            cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)
        return pause, process_same_image, display_new_image, surface_list, pos_list, cam, name

def CAM_pygame_loop(font, webcam, model, display, target_layers, class_list, window_size):
    surface_list = []
    pos_list = []
    model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    img = webcam.get_image()
    image_size = [img.get_width(), img.get_height()]
    offset_list, window_size = get_offset_list(window_size, image_size)
    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)
    process_same_image = False
    display_new_image = False
    pause = False
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_m:
                    print("M pressed while NOT pause")
                    surface_list.clear()
                    pos_list.clear()
                    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)

                if event.key == K_SPACE:
                    pause = True
                    # obtain detection data
                    probabilities = get_probabilities(model, img)
                    probabilities = probabilities.to('cpu')
                    target_class = np.argmax(probabilities.data.numpy())
                    class_name = class_list[target_class]
                    class_score = probabilities[target_class]
                    full_name = name + ': '+class_name+'| Score: '+str(round(float(class_score)*100,2))+'%'
                    print(full_name)  

                    t0 = time.time()
                    # get the cam heat map in a pygame image
                    cam_surface = roc_functions.surface_to_cam(img, cam)
                    print('time needed for visualization method creation :', time.time()-t0)
                    display.blit(cam_surface,offset_list[pos])
                    text = font.render(full_name, True, WHITE)
                    text_pos = offset_list[pos]
                    print(text_pos)
                    display.blit(text,text_pos)
                    pygame.display.flip()
            
                    while pause:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                sys.exit()
                            elif event.type == pygame.KEYDOWN:
    
                                if event.key == K_SPACE:
                                    print("Space pressed")
                                    pause = False
                                    process_same_image = False
                                    break
                                if event.key == K_ESCAPE:
                                    pygame.quit()
                                    sys.exit()
                                if event.key == K_m:
                                    print("M pressed while pause")
                                    process_same_image = True
                                    pause = False
                                    surface_list.clear()
                                    pos_list.clear()
                                    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)
                                    break
                                if event.key == K_n:
                                    print("N pressed while pause")
                                    surface_list.append(cam_surface)
                                    pos_list.append(offset_list[pos])
                                    process_same_image = True
                                    add_new_image = True
                                    cam, name, pos = roc_functions.method_menu(font, display, model, target_layers)
                                    break

                        if process_same_image: break
                        
        if process_same_image and add_new_image:
            t0 = time.time()
            cam_surface = roc_functions.surface_to_cam(img, cam)
            # Store the new cam
            surface_list.append(cam_surface)
            pos_list.append(offset_list[pos])
            print('time needed for visualization method creation :', time.time()-t0)
            for offset in offset_list:
                if offset in pos_list: # to display the stored cams
                    idx = pos_list.index(offset)
                    display.blit(surface_list[idx], offset)
                    text = font.render(full_name, True, WHITE)
                    display.blit(text,offset)
                else: # to display the rest of images
                    display.blit(img,offset)
            
            text = font.render(full_name, True, WHITE)
            display.blit(text,offset_list[pos])
            pygame.display.flip()
            add_new_image = False

        if not process_same_image and not pause:
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