import time
from adafruit_pca9685 import PCA9685

#PCA9685 PWM controller
pca = PCA9685()
pca.frequency = 50  # Set PWM frequency to 50 Hz

#PWM channel where your servo is connected
servo_channel = 0

# Function to set the servo angle using PWM with time control
def set_servo_angle_with_time(channel, angle, duration):
    start_time = time.time()
    initial_angle = pca.channels[channel].duty_cycle / 0xFFFF * 180.0

    while True:
        elapsed_time = time.time() - start_time
        progress = min(1.0, elapsed_time / duration)

        current_angle = initial_angle + (angle - initial_angle) * progress
        set_servo_angle(channel, current_angle)

        if progress >= 1.0:
            break

        time.sleep(0.01)

# Function to set the servo angle using PWM
def set_servo_angle(channel, angle):
    # Calculate the PWM duty cycle based on the servo angle
    duty_cycle = int((angle / 180.0) * (0xFFFF))
    pca.channels[channel].duty_cycle = duty_cycle

# Main code
try:
    while True:

        calculated_angle = 60

        # Set the servo angle over a specified duration (in seconds)
        set_servo_angle_with_time(servo_channel, calculated_angle, duration=2.0)

        # You can perform additional actions or wait as needed

except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C)
    pca.deinit()

    print("\nProgram terminated")
