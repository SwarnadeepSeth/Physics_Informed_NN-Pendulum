import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Parameters
L = 0.025  # Length of pendulum
g = 9.81  # Acceleration due to gravity

# ODE function for a simple pendulum
def simple_pendulum_eqn(state, t, L, g):
    theta, theta_dot = state
    theta_ddot = -g / L * np.sin(theta)
    return [theta_dot, theta_ddot]

# Generate time values
t = np.linspace(0, 10, 1000)

# Initial state [theta, theta_dot]
initial_state = [np.pi/4, 0]

# Numerical solution of the simple pendulum ODEs
states = odeint(simple_pendulum_eqn, initial_state, t, args=(L, g))
theta_values = states[:, 0]

# Calculate positions of the pendulum bob
x_values = L * np.sin(theta_values)
y_values = -L * np.cos(theta_values)

# Create a figure and axis
fig, ax = plt.subplots()

# Set axis limits
ax.set_xlim(-1.5 * L, 1.5 * L)
ax.set_ylim(-1.5 * L, 0.5 * L)

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Simple Pendulum Animation')

# Create line for the pendulum
line, = ax.plot([], [], 'o-', lw=2, label='Pendulum')
trajectory_line, = ax.plot([], [], 'b-', lw=1, alpha=0.5)  # Trajectory of the pendulum bob
lines = [line, trajectory_line]

# Create a legend
ax.legend()

# Initialization function
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Animation function
def animate(i):
    x = x_values[i]
    y = y_values[i]

    line.set_data([0, x], [0, y])
    trajectory_line.set_data(x_values[:i+1], y_values[:i+1])  # Trace the trajectory of the pendulum bob
    
    return lines

# Create the animation
anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=20, blit=True)

# Display the animation
plt.show()

# save the animation as gif
anim.save('simple_pendulum.gif', writer='imagemagick', fps=60)

 

plt.plot(t, theta_values)
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.title('Angle vs Time')
plt.show()