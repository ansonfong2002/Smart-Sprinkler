import os
import sys
from source.system import Squirt 

def process_config(dir):
    config = []
    with open(f"{dir}/config/system.config", "r") as file:
        for line in file:
            config.append(int(line.split()[1]))
    
    return config

# start system
def start(config):
    # set up device pins
    device = Squirt(config)

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
    start(config)

    # clean up locks
    os.remove(".lgd-nfy0")