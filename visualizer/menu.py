"""
@author: Pablo Roca - github.com/RocaPiedra
@original author: Utku Ozbulak - github.com/utkuozbulak
"""
from time import sleep
from cv2 import *
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
        while not parameters.activate_deleter:
            ret, frame = cap.read()
            if not ret:
                break
            cam.visualization_pipeline(frame, None, True)

    elif option == 2:
        print('Image selected as input')
        path = '../input_images/carla_input/'
        image_paths = get_image_path(path,None)
        # do a function for target class
        for paths in image_paths:
            original_image = cv2.imread(paths)
            # take the name of the file without extension:
            file_name_to_export = os.path.splitext(ntpath.basename(paths))[0]
            # process the image and obtain the overlayed with cam method
            cv2_heatmap_on_image = cam.visualization_pipeline(original_image, None, True)
            path_to_file = os.path.join('../results', file_name_to_export+'_'+cam.method_name+'_'+pretrained_model.__class__.__name__+'.png')
            save_image(cv2_heatmap_on_image, path_to_file)
            print(f'{cam.method_name} completed for image:',file_name_to_export)

    elif option == 3:
        print('Video selected as input')
        frame_counter = 0
        video_path = '../input_images/video_input/Road-traffic.mp4'
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened() and not parameters.activate_deleter:
            ret, frame = cap.read()
            frame_counter += 1
            if not ret:
                break
            cam.visualization_pipeline(frame, None, True)
    
    elif option == 4:
        print('Carla selected as input')
        subp_unreal, subp_traffic = launch_carla_simulator_locally()
        platform = sensor_platform()
        sensor = platform.set_sensor()
        sensor.listen(lambda data: cam.visualization_pipeline(data, platform, True, False))
        while not parameters.activate_deleter:sleep(2)
        
        subp_unreal.terminate()
        subp_traffic.terminate()

        del platform
        exit()
    else:
        print('Wrong option, shutting down application...')
        exit()
