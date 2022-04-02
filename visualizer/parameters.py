import os

visualize = False # prints information during execution of layers and steps to debug
sendToGPU = True
activate_deleter = False #global variable to close cleanly the application
if os.name == 'nt':
    unreal_engine_path = r"C:\Users\pablo\CARLA_0.9.13\Carla\CarlaUE4.exe"
else:
    unreal_engine_path = '/opt/carla-simulator/CarlaUE4.sh'