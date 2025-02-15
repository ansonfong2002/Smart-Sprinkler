# VALVE
from gpiozero import PWMOutputDevice
from gpiozero import OutputDevice
import datetime
import os
import sys

valve = PWMOutputDevice(12)
valve.frequency = 7500

try:
    while(True):
        val = float(input("Enter PWM Range 0-1: "))
        if (val <= 1 and val >= 0):
            valve.value = val
            print(valve.value)

except KeyboardInterrupt:
    valve.close()
    os.remove(".lgd-nfy0")
    sys.exit()