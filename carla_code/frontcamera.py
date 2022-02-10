#!/usr/bin/env python3

# authors: RocaPiedra
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
from pygame import key
import numpy as np

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla

#*******************************************************************************
#%%
IM_WIDTH = 640
IM_HEIGHT = 480
IM_FOV = 100
IM_SHUTTER = 30

actor_list = []
sensor_list = []
# to RGB avoiding alpha
def process_img(image):
    i = np.array(image.raw_data)
    print(i.shape)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    print(i2.shape)
    i3 = i2[:,:,:3] #only get rgb avoids alpha 
    cv2.imshow("",i3)
    cv2.waitKey(1)
    return i3/255.0

def main():
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(6.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        
        bp = blueprint_library.filter("mustang")[0]
        print(bp)
        
        spawn_point = random.choice(world.get_map().get_spawn_points())
        print(spawn_point)
        
        vehicle = world.spawn_actor(bp, spawn_point)
        vehicle.set_autopilot(True)
        # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        actor_list.append(vehicle)

        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        cam_bp.set_attribute("fov", f"{IM_FOV}")
        cam_bp.set_attribute("shutter_speed", f"{IM_SHUTTER}")

        spawn_point = carla.Transform(carla.Location(x=1, z=1.3))
        sensor = world.spawn_actor(cam_bp, spawn_point, attach_to=vehicle)
        sensor.listen(lambda data: process_img(data))
        sensor_list.append(sensor)

        while 1:
            time.sleep(10)
    finally:
        for actor in actor_list:
            print(actor)
            actor.destroy()
        for sensors in sensor_list:
            print(sensors)
            sensors.destroy()
        print("all actors destroyed")

if __name__ == "__main__":
    pygame.init()
    main()