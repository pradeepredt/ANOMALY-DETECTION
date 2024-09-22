import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def simulate_data_stream(size=1000, seasonality=100, noise=0.5, drift=0.005):
    # Validate input parameters
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")
    if not isinstance(seasonality, int) or seasonality <= 0:
        raise ValueError("Seasonality must be a positive integer.")
    if not isinstance(noise, (int, float)) or noise < 0:
        raise ValueError("Noise must be a non-negative number.")
    if not isinstance(drift, (int, float)):
        raise ValueError("Drift must be a number.")
    
    # Create time series
    time = np.arange(size)
    
    # Seasonal component using sine wave
    seasonal = np.sin(2 * np.pi * time / seasonality)
    
    # Drift (trend) added to simulate gradual increase
    trend = drift * time
    
    # Add random noise to data
    random_noise = np.random.normal(0, noise, size=size)
    
    # Introduce anomalies at random positions
    anomaly_indices = random.sample(range(50, size-50), k=10)
    anomaly_magnitude = [random.uniform(2, 4) for _ in range(10)]
    
    # Combine all components to create data stream
    data = trend + seasonal + random_noise
    for i, idx in enumerate(anomaly_indices):
        data[idx] += anomaly_magnitude[i]  # Add anomaly magnitude to data
    
    return data

def plot_data(data):
    # Validate data before plotting
    if not isinstance(data, np.ndarray):
        raise ValueError("Data must be a NumPy array.")
    if data.ndim != 1:
        raise ValueError("Data must be a 1D array.")
    
    # Plot the data stream
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Data Stream', color='blue')
    plt.title('Simulated Data Stream with Seasonality and Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

class AnomalyFinder:
    def __init__(self, window=50, threshold=2):
        # Initialize window size and threshold for anomaly detection
        self.window = window
        self.threshold = threshold
        self.window_data = []
        self.ema = 0
        self.ema_sq = 0  # For calculating standard deviation

    def update_stats(self, value):
        # Update EMA and square of EMA for standard deviation
        alpha = 2 / (self.window + 1)
        if len(self.window_data) < self.window:
            self.window_data.append(value)
            self.ema = np.mean(self.window_data)
            self.ema_sq = np.mean(np.square(self.window_data))
        else:
            self.window_data.pop(0)  # Remove oldest data
            self.window_data.append(value)  # Add new value
            self.ema = alpha * value + (1 - alpha) * self.ema
            self.ema_sq = alpha * (value ** 2) + (1 - alpha) * self.ema_sq
    
    def detect_anomaly(self, value):
        # Check if the value is an anomaly
        if len(self.window_data) < self.window:
            return False
        std_dev = np.sqrt(self.ema_sq - self.ema ** 2)  # Calculate standard deviation
        z_score = (value - self.ema) / (std_dev + 1e-5)  # Calculate Z-score
        return abs(z_score) > self.threshold  # Flag as anomaly if Z-score exceeds threshold

def visualize_real_time(data, detector):
    # Set up the figure for real-time visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_data, y_data = [], []
    anomaly_x, anomaly_y = [], []

    ax.set_xlim(0, len(data))
    ax.set_ylim(min(data) - 1, max(data) + 1)
    
    line, = ax.plot([], [], color='blue', label='Data Stream')
    anomaly_scatter = ax.scatter([], [], color='red', label='Anomalies')

    plt.title('Real-Time Data Stream with Anomaly Detection')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(loc='upper right')

    def init():
        # Initialize empty plot
        line.set_data([], [])
        anomaly_scatter.set_offsets(np.empty((0, 2)))
        return line, anomaly_scatter

    def update(frame):
        # Update plot for each frame
        x_data.append(frame)
        y_data.append(data[frame])
        
        detector.update_stats(data[frame])  # Update statistics with current value
        is_anomaly = detector.detect_anomaly(data[frame])  # Check for anomaly
        
        if is_anomaly:
            anomaly_x.append(frame)  # Record anomaly index
            anomaly_y.append(data[frame])  # Record anomaly value
        
        line.set_data(x_data, y_data)  # Update line data
        
        if anomaly_x and anomaly_y:
            anomaly_scatter.set_offsets(np.c_[anomaly_x, anomaly_y])  # Update anomaly scatter
        
        return line, anomaly_scatter

    ani = FuncAnimation(fig, update, frames=range(len(data)), init_func=init,
                        blit=True, interval=100, repeat=False)
    
    plt.show()

if __name__ == "__main__":
    try:
        data_stream = simulate_data_stream()  # Generate data stream
        plot_data(data_stream)  # Plot initial data
        detector = AnomalyFinder()  # Initialize anomaly detector
        visualize_real_time(data_stream, detector)  # Visualize data in real-time
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
