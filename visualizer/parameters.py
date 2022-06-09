import os

visualize = False # prints information during execution of layers and steps to debug
sendToGPU = True
call_pause = False
activate_deleter = False # global variable to close cleanly the application
fixed_delta_seconds = 0.05 # to maintain constant delta time in every client
if os.name == 'nt':
    unreal_engine_path = r"C:\Users\pablo\CARLA_0.9.13\Carla\CarlaUE4.exe"
else:
    unreal_engine_path = '/opt/carla-simulator/CarlaUE4.sh'
WHITE = (255, 255, 255)
BUTTON_COLOR = (169,169,169)