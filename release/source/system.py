from gpiozero import PWMOutputDevice
from gpiozero import OutputDevice
import time

# SQUIRT system class
class Squirt:
    # initialize pins, vars, log
    def __init__(self, config):
        self.valve = PWMOutputDevice(config[0])
        self.pulse = OutputDevice(config[1])
        self.direction = OutputDevice(config[2])
        self.valve.frequency = config[3]
        self.stops = config[4]
        self.steps_per_rev = config[5]
        self.log = [f"VALVE PIN {config[0]}"]
        self.log.append(f"PULSE PIN {config[1]}")
        self.log.append(f"DIR PIN {config[2]}")
        self.log.append(f"FREQ {config[3]}")
        self.log.append(f"STOPS {config[4]}")
        self.log.append(f"STEPS PER REV {config[5]}")

        self.set_valve_pos(0)

    # translate distance to valve position (MAX 1.0)
    def normalize(self, value):
        # linear regression model, R2 = 0.9943
        #return (value * 0.0006253) + 0.1446
        return (value * 0.0006908) + 0.1706

    # sets valve position
    def set_valve_pos(self, position):
        if (position <= 1 and position >= 0):
            self.valve.value = position
            self.log.append(f"VALVE POS {position}")

    # limit in measurement unit: translation layer inside function
    def valve_cycle(self, limit):
        normalized = self.normalize(limit)
        self.log.append(f"VALVE CYL {normalized}")
        pos = 0.2
        while pos < normalized:
            self.set_valve_pos(pos)
            time.sleep(4)
            pos += 0.01

        self.set_valve_pos(0)
        time.sleep(5)

    # toggle direction
    def toggle_direction(self):
        self.direction.toggle()
        self.log.append(f"DIR SET {self.direction.value}")

    # perform STEPS PER REV / NUM STEPS steps
    def rotate(self):
        steps = int(self.steps_per_rev / self.stops)
    
        for i in range(steps):
            self.pulse.on()
            time.sleep(0.001)
            self.pulse.off()
            time.sleep(0.0005)
        self.log.append(f"PULSE {steps}")

    # get log
    def get_log(self): return self.log

    # clean log
    def clear_log(self): self.log = []