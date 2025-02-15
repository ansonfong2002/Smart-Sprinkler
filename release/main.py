import os
import sys
import source.depth
from source.system import Squirt 

def process_config(dir):
    config = []
    with open(f"{dir}/config/system.config", "r") as file:
        for line in file:
            config.append(int(line.split()[1]))
    
    return config

# start system
def start(config, dir):
    # set up device pins
    device = Squirt(config)

    # test run:
        # take snapshot
        # run depth processing
        # send value to device.valve_cycle
        # device.rotate
    os.system(f"libcamera-still --camera 0 -o {dir}/temp/left.JPEG")
    os.system(f"libcamera-still --camera 1 -o {dir}/temp/right.JPEG")
    distance = depth.process_imgs(dir)
    print(f"Distance to boundary: {distance:.2f} cm")

    #device.valve_cycle(distance)

    # for each stop in num stops
        # take snapshots, load into temp
        # run depth processing
        # send value to device.valve_cycle
        # device.rotate
    # upon completion, toggle direction and spin back

    # get log
    device_log = device.get_log()
    print(device_log)

# main
if __name__ == "__main__":
    # get parent path and process configs
    dir = os.path.dirname(os.path.abspath(__file__))
    config = process_config(dir)

    # start system
    start(config, dir)

    # clean up locks
    os.remove(".lgd-nfy0")