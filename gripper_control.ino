#include <Servo.h>

// Create a Servo object
Servo gripperServo;

// Specify the pin where your servo is connected
const int servoPin = 9;

// Function to set the servo angle with time control
void setServoAngleWithTime(int angle, int duration) {
  int initialAngle = gripperServo.read();
  unsigned long startTime = millis();

  while (true) {
    unsigned long elapsedTime = millis() - startTime;
    float progress = min(1.0, float(elapsedTime) / duration);

    int currentAngle = initialAngle + int(float(angle - initialAngle) * progress);
    gripperServo.write(currentAngle);

    if (progress >= 1.0) {
      break;
    }

    delay(10);  // Adjust the delay time based on your needs
  }
}

void setup() {
  // Attach the servo to the specified pin
  gripperServo.attach(servoPin);
}

void loop() {
  // Your calculation logic to determine the desired servo angle
  // For example, you might have a variable like calculatedAngle
  int calculatedAngle = 60;

  // Set the servo angle over a specified duration (in milliseconds)
  setServoAngleWithTime(calculatedAngle, 2000);  // 2000 milliseconds = 2 seconds

  // You can perform additional actions or wait as needed
  delay(2000);  // 2000 milliseconds = 2 seconds
}
