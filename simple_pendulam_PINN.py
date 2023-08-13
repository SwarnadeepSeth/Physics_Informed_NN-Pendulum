import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import matplotlib.pyplot as plt


# Parameters
L = 1.0  # Length of pendulum
g = 9.81  # Acceleration due to gravity

# Generate training data
num_samples = 1000
theta_initial = np.random.uniform(-np.pi, np.pi, num_samples)
theta_dot_initial = np.random.uniform(-5, 5, num_samples)

def simulate_simple_pendulum(theta_initial, theta_dot_initial):
    # Initialize arrays to store data
    theta_values = np.zeros((len(theta_initial), len(t)))
    theta_dot_values = np.zeros((len(theta_initial), len(t)))

    # Loop over each set of initial conditions
    for i in range(len(theta_initial)):
        # Initial conditions
        theta = theta_initial[i]
        theta_dot = theta_dot_initial[i]

        # Time values and time step
        t_eval = np.linspace(0, 10, len(t))
        dt = t_eval[1] - t_eval[0]

        # Arrays to store theta and theta_dot over time
        theta_values_single = np.zeros_like(t_eval)
        theta_dot_values_single = np.zeros_like(t_eval)

        # Numerical integration using Euler's method
        for j in range(len(t_eval)):
            theta_values_single[j] = theta
            theta_dot_values_single[j] = theta_dot

            # Compute angular acceleration (second derivative)
            theta_double_dot = -g / L * np.sin(theta)

            # Update angular velocity and angle using Euler's method
            theta_dot += theta_double_dot * dt
            theta += theta_dot * dt

        theta_values[i] = theta_values_single
        theta_dot_values[i] = theta_dot_values_single

    return theta_values, theta_dot_values


# Generate new initial conditions for prediction
num_predictions = 10
new_theta_initial = np.random.uniform(-np.pi, np.pi, num_predictions)
new_theta_dot_initial = np.random.uniform(-5, 5, num_predictions)

# Simulate the simple pendulum to get ground truth data for predictions
ground_truth_theta, ground_truth_theta_dot = simulate_simple_pendulum(new_theta_initial, new_theta_dot_initial)






# Define the Physics-Informed Neural Network architecture
class PendulumPINN(tf.keras.Model):
    def __init__(self):
        super(PendulumPINN, self).__init__()
        self.dense1 = layers.Dense(32, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.output_layer = layers.Dense(2)  # Output: theta, theta_dot

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        outputs = self.output_layer(x)
        return outputs

# Instantiate the PINN model
model = PendulumPINN()

# Define loss functions
data_loss_fn = losses.MeanSquaredError()

def physics_loss_fn(theta_initial, theta_dot_initial, predicted_theta, predicted_theta_dot):
    # Convert predicted_theta and predicted_theta_dot to numpy arrays
    predicted_theta = predicted_theta.numpy()
    predicted_theta_dot = predicted_theta_dot.numpy()
    
    # Compute the time derivative of predicted_theta_dot
    predicted_theta_double_dot = -g / L * np.sin(predicted_theta)
    
    # Compute the physics loss using mean squared error
    physics_loss = np.mean((predicted_theta_dot - predicted_theta_double_dot) ** 2)
    
    return physics_loss

# Define optimizer
optimizer = optimizers.Adam(learning_rate=0.001)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        # Forward pass
        predicted_theta, predicted_theta_dot = model(tf.convert_to_tensor([theta_initial, theta_dot_initial], dtype=tf.float32))
        
        # Compute loss
        data_loss = data_loss_fn(ground_truth_theta, predicted_theta) + data_loss_fn(ground_truth_theta_dot, predicted_theta_dot)
        physics_loss = physics_loss_fn(theta_initial, theta_dot_initial, predicted_theta, predicted_theta_dot)
        total_loss = data_loss + physics_loss

    # Compute gradients and update model parameters
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Prediction and visualization
# Predict using the trained PINN
predicted_theta, predicted_theta_dot = model(tf.convert_to_tensor([new_theta_initial, new_theta_dot_initial], dtype=tf.float32))

# Plot the predicted and ground truth trajectories
plt.figure(figsize=(10, 6))

for i in range(num_predictions):
    plt.plot(t, predicted_theta[i], label=f'Prediction {i + 1}', linestyle='dashed')
    plt.plot(t, ground_truth_theta[i], label=f'Ground Truth {i + 1}')

plt.xlabel('Time')
plt.ylabel('Angular Displacement (Theta)')
plt.title('Predicted and Ground Truth Trajectories')
plt.legend()
plt.grid()

plt.show()

 
