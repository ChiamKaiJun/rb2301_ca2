"""Small scalar PID controller. No ROS, waypoint, or simulator dependencies."""
import math


class PIDController:
    def __init__(self, kp, ki, kd, output_limit, integral_limit=1.0, angular=False):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = abs(output_limit)
        self.integral_limit = abs(integral_limit)
        self.angular = angular
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def compute(self, error, dt):
        # First sample / paused clock: P remains available, I and D do not advance.
        derivative = 0.0
        candidate = self.integral
        if dt > 0.0:
            if self.ki != 0.0:
                candidate = max(-self.integral_limit,
                                min(self.integral_limit, self.integral + error * dt))
            if self.prev_error is not None:
                change = error - self.prev_error
                if self.angular:  # Wrap changes across +/-pi before differentiation.
                    change = (change + math.pi) % (2.0 * math.pi) - math.pi
                derivative = change / dt
        output = self.kp*error + self.ki*candidate + self.kd*derivative
        if abs(output) <= self.output_limit or output * (self.ki*error) < 0.0:  # Anti-windup.
            self.integral = candidate
        else:
            output = self.kp*error + self.ki*self.integral + self.kd*derivative
        self.prev_error = error
        return max(-self.output_limit, min(self.output_limit, output))  # Saturate final output.
