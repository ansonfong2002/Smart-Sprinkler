import os
import subprocess
import sys
import time
import source.depth as depth
from source.system import Squirt 

def process_config(dir):
    config = []
    with open(f"{dir}/config/system.config", "r") as file:
        for line in file:
            config.append(int(line.split()[1]))
    
    return config

def motor_tune():
    device = Squirt(config)
    while True:
        ready = input("<> Enter T to toggle direction, enter to rotate... ")
        if ready == "T":
            device.toggle_direction()
        else:
            device.rotate()


# start system
def start(config, dir):
    # set up device pins
    device = Squirt(config)

    ready = input("<> Initialized, enter any key to start...\n")

    print("<> Capturing stereo images...")
    subprocess.run(f"libcamera-still --camera 0 -o {dir}/temp/left.JPEG", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(" > Left capture done")
    subprocess.run(f"libcamera-still --camera 1 -o {dir}/temp/right.JPEG", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(" > Right capture done")
    
    distance = depth.process_imgs(dir)
    print(f" > Distance to boundary: {distance:.2f} cm")

    # for each stop in num stops
        # take snapshots, load into temp
        # run depth processing, save in cv
        # send value to device.valve_cycle
        # device.rotate
    # upon completion, toggle direction and spin back

    # get log
    device_log = device.get_log()
    with open("log.txt", 'a') as logfile:
        for entry in device_log:
            logfile.write(entry)
            logfile.write("\n")
    device.clear_log()

# main
if __name__ == "__main__":
    # get parent path and process configs
    dir = os.path.dirname(os.path.abspath(__file__))
    config = process_config(dir)

    # start system
    try: 
        #motor_tune()
        start(config, dir)
    except KeyboardInterrupt: pass
    
    # clean up locks
    print("\n<> System terminating\n")
    os.remove(".lgd-nfy0")