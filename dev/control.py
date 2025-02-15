# MOTOR DRIVER
# pulse+ = GPIO16
# dir+ = GPIO21

# VALVE
# plug in all wires to LEDs if necessary
# green = GPIO12

from gpiozero import PWMOutputDevice
from gpiozero import OutputDevice
import time

# pulse = OutputDevice(16)
# direction = OutputDevice(21)
# 
# print("Initializing step")
# direction.on()
# time.sleep(0.5)
# for i in range(10):
#     print("Initializing control loop")
#     for i in range(40):
#         pulse.on()
#         time.sleep(0.001)
#         pulse.off()
#         time.sleep(0.0005)
#     time.sleep(1)
# 
# print("Performed 400 steps")

valve = PWMOutputDevice(12)

for i in range(10):
    valve.value = i/10
    time.sleep(1)