"""
@author: Pablo Roca - github.com/RocaPiedra
@original author: Utku Ozbulak - github.com/utkuozbulak
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

from misc_functions import save_class_activation_images, apply_colormap_on_image
from roc_functions import *
from gradcam_visualizer import GradCam
from scorecam_visualizer import ScoreCam

sys.path.append('..')

from carlacomms.carla_sensor_platform import sensor_platform
# code options
import parameters

if __name__ == '__main__':
    #Use input Arguments
    input_arguments = sys.argv
    # Choose model
    if len(input_arguments) >= 2:
        if len(input_arguments[1])==1:
            pretrained_model = choose_model(input_arguments[1], model_name = None)
        elif len(input_arguments[1])>1:
            pretrained_model = choose_model(0, input_arguments[1])
    else:
        pretrained_model = choose_model()
    if torch.cuda.is_available() and parameters.sendToGPU:
        if parameters.visualize: print('CUDA ENABLED in main')
        pretrained_model.to('cuda')
        if parameters.visualize: print(f'is model loaded to gpu? -> {next(pretrained_model.parameters()).is_cuda}')
    elif not parameters.sendToGPU:
        print('Using CPU') 
    else:
        print('CUDA NOT ENABLED in main, exit')
        exit()
    # GradCAM object declaration
    option = int(input('What method do you want to use: \n1.GradCAM\n2.ScoreCAM\n'))
    if option == 2:
        cam = ScoreCam(pretrained_model, target_layer=None)
    else:
        cam = GradCam(pretrained_model, target_layer=None)

    # Choose input option
    if len(input_arguments) >= 3:    
        if int(input_arguments[2]) in (1,2,3,4):
            option = int(input_arguments[2])
    else:
        option = int(input('What type of input do you want: \n1.Webcam\n2.Image Folder\n3.Video \n4.Carla Simulator\n'))
        
    if option == 1:
        print('Webcam selected as input')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # /dev/video0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            original_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('Input',frame)
            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print(f'Closing {cam.method_name}, shutting down application...')
                cap.release()
                cv2.destroyAllWindows()
                exit()

            prep_img = preprocess_image(original_image, parameters.sendToGPU)
            cam_gen = cam.generate_cam(prep_img)
            # Show mask
            heatmap, heatmap_on_image = apply_colormap_on_image(original_image, cam_gen, 'hsv')
            cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
            cv2.imshow(f'{cam.method_name}',cv2_heatmap_on_image)
            if c == 27:
                print(f'Closing {cam.method_name}, shutting down application...')
                cap.release()
                cv2.destroyAllWindows()
                exit()

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
            cam_gen = cam.generate_cam(prep_img)
            # Save mask
            save_class_activation_images(original_image, cam_gen, file_name_to_export, pretrained_model.__class__.__name__)
            print(f'{cam.method_name} completed for image:',file_name_to_export)

    elif option == 3:
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
            cv2.imshow('Input',frame)
            
            prep_img = preprocess_image(original_image)
            file_name_to_export = f'webcam_{frame_counter}'
            cam_gen = cam.generate_cam(prep_img)
            # Show mask
            heatmap, heatmap_on_image = apply_colormap_on_image(original_image, cam_gen, 'hsv')
            cv2_heatmap_on_image = cv2.cvtColor(np.array(heatmap_on_image), cv2.COLOR_RGB2BGR)
            cv2.imshow(f'{cam.method_name}',cv2_heatmap_on_image)

            c = cv2.waitKey(1) # ASCII 'Esc' value
            if c == 27:
                print(f'Closing {cam.method_name}, shutting down application...')
                cap.release()
                cv2.destroyAllWindows()
                exit()
    
    elif option == 4:
        print('Carla selected as input')
        subp_unreal, subp_traffic = launch_carla_simulator_locally()
        platform = sensor_platform()
        sensor = platform.set_sensor()
        sensor.listen(lambda data: cam.visualization_pipeline(data, platform,True))
        while not parameters.activate_deleter:sleep(2)
        
        subp_unreal.terminate()
        subp_traffic.terminate()

        del platform
        exit()
    else:
        print('Wrong option, shutting down application...')
        exit()
