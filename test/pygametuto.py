import numpy as np
import sys

sys.path.append('../visualizer')

from gradcam_visualizer import GradCam
from scorecam_visualizer import ScoreCam
import roc_functions

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
# import torch
import numpy as np
import time

try:
    import pygame
    import pygame.camera
    import pygame.image
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_q
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

def main():
    pygame.init()
    pygame.font.init() #for fonts rendering
    pygame.camera.init()

    font = pygame.font.SysFont(None, 24)
    gradcam_text = font.render('GradCAM', True, WHITE)
    scorecam_text = font.render('ScoreCAM', True, WHITE)
    ablationcam_text = font.render('AblationCAM', True, WHITE)
    xgradcam_text = font.render('XGradCAM', True, WHITE)
    eigencam_text = font.render('EigenCAM', True, WHITE)
    fullgrad_text = font.render('FullGrad', True, WHITE)

    
    cameras = pygame.camera.list_cameras()
    res = '1920x1080'
    window_size = [int(x) for x in res.split('x')]
    webcam = pygame.camera.Camera(cameras[0])
    pause = False
    
    model = resnet18(pretrained=True)
    target_layers = [model.layer4[-1]]
    
    grad = False
    score = False
    ablation = False
    xgrad = False
    eigen = True
    full = True

    if grad:
        gradcam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=True)
    if score:
        scorecam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=True)
    if ablation:
        ablationcam = AblationCAM(model=model, target_layers=target_layers, use_cuda=True)
    if xgrad:
        xgradcam = XGradCAM(model=model, target_layers=target_layers, use_cuda=True)
    if eigen:
        eigencam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
    if full:
        fullgrad = FullGrad(model=model, target_layers=target_layers, use_cuda=True)
    
    targets = ClassifierOutputTarget(281)

    webcam.start()
    img = webcam.get_image()
    image_size = [img.get_width(), img.get_height()]
    offset_list, window_size = get_offset_list(window_size, image_size)
    display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("PyGame Camera View")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == K_SPACE:
                    pause = True

                    if grad:
                        t0 = time.time()
                        gradcam_surface = roc_functions.surface_to_cam(img, gradcam)
                        print('time needed for GradCAM creation :', time.time()-t0)
                        display.blit(gradcam_surface,offset_list[0])
                        display.blit(gradcam_text,offset_list[0])
                        pygame.display.flip()
                    if score:
                        t0 = time.time()
                        scorecam_surface = roc_functions.surface_to_cam(img, scorecam)
                        print('time needed for ScoreCAM creation :', time.time()-t0)
                        display.blit(scorecam_surface,offset_list[2])
                        display.blit(scorecam_text,offset_list[2])
                        pygame.display.flip()
                    if ablation:
                        t0 = time.time()
                        ablationcam_surface = roc_functions.surface_to_cam(img, ablationcam)
                        print('time needed for AblationCAM creation :', time.time()-t0)
                        display.blit(ablationcam_surface,offset_list[4])
                        display.blit(ablationcam_text,offset_list[4])
                        pygame.display.flip()
                    if xgrad:
                        t0 = time.time()
                        xgradcam_surface = roc_functions.surface_to_cam(img, xgradcam)
                        print('time needed for xgradcam creation :', time.time()-t0)
                        display.blit(xgradcam_surface,offset_list[1])
                        display.blit(xgradcam_text,offset_list[1])
                        pygame.display.flip()
                    if eigen:
                        t0 = time.time()
                        eigencam = EigenCAM(model=model, target_layers=target_layers, use_cuda=True)
                        eigencam_surface = roc_functions.surface_to_cam(img, eigencam)
                        print('time needed for eigencam creation :', time.time()-t0)
                        display.blit(eigencam_surface,offset_list[3])
                        display.blit(eigencam_text,offset_list[3])
                        pygame.display.flip()
                    if full:
                        t0 = time.time()
                        fullgrad = FullGrad(model=model, target_layers=target_layers, use_cuda=True)
                        fullgrad_surface = roc_functions.surface_to_cam(img, fullgrad)
                        print('time needed for fullgrad creation :', time.time()-t0)
                        display.blit(fullgrad_surface,offset_list[5])
                        display.blit(fullgrad_text,offset_list[5])
                        pygame.display.flip()

                    
                    while pause:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                sys.exit()
                            elif event.type == pygame.KEYDOWN:
                                if event.key == K_SPACE:
                                    pause = False

        for offset in offset_list:
            display.blit(img,offset)
        pygame.display.flip()
        img = webcam.get_image()
    

if __name__ == '__main__':
    main()