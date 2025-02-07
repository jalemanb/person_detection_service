class AdaptiveThr:
    def __init__(self, initial_value: float, process_noise: float, measurement_noise: float):
        """
        Initialize the Kalman Filter for a single parameter.
        :param initial_value: Initial estimate of the parameter
        :param process_noise: Process noise variance (Q)
        :param measurement_noise: Measurement noise variance (R)
        """
        self.x = initial_value  # Initial state estimate
        self.P = 1.0  # Initial covariance estimate
        self.Q = process_noise  # Process noise covariance
        self.R = measurement_noise  # Measurement noise covariance
        self.F = 1.0  # State transition coefficient
        self.H = 1.0  # Measurement coefficient

    def predict(self):
        """
        Predict the next state and update the covariance.
        """
        self.x = self.F * self.x  # State prediction (no external input)
        self.P = self.F * self.P * self.F + self.Q  # Covariance prediction

    def update(self, measurement: float):
        """
        Update the filter with a new measurement.
        :param measurement: New measurement value
        """
        y = measurement - (self.H * self.x)  # Measurement residual
        S = self.H * self.P * self.H + self.R  # Innovation covariance
        K = self.P * self.H / S  # Kalman gain
        self.x = self.x + K * y  # Updated state estimate
        self.P = (1 - K * self.H) * self.P  # Updated covariance estimate

    def get_estimate(self):
        """
        Get the estimated state and covariance.
        :return: Tuple (estimated_value, covariance)
        """
        return self.x, self.P

# Example usage:
# kf = AdaptiveThr(initial_value=0.0, process_noise=0.01, measurement_noise=0.1)
# kf.predict()
# kf.update(1.0)
# print(kf.get_estimate())
