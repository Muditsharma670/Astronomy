import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

# --- Constants & Units ---
# Units: Distance = AU, Time = Years, Mass = Solar Masses
# Gravitational Constant in these units: G = 4 * pi^2
G = 4 * np.pi**2

class Body:
    def __init__(self, name, mass, pos, vel, color):
        self.name = name
        self.mass = mass
        self.pos = np.array(pos, dtype=float)  # x, y, z in AU
        self.vel = np.array(vel, dtype=float)  # vx, vy, vz in AU/Year
        self.color = color
        self.trajectory = np.array([pos])      # History for plotting trails

def compute_accelerations(bodies, positions):
    """
    Calculates the acceleration for every body based on Newton's Law of Gravitation.
    F = G * m1 * m2 / r^2
    a = F / m1 = G * m2 / r^2
    """
    n = len(bodies)
    accelerations = np.zeros((n, 3))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Vector from body i to body j
                r_vec = positions[j] - positions[i]
                # Distance magnitude
                r_mag = np.linalg.norm(r_vec)
                
                # Newton's Law: a_i = G * m_j * (r_vec / r^3)
                # We calculate force contribution of body j on body i
                acc = G * bodies[j].mass * (r_vec / (r_mag**3))
                accelerations[i] += acc
    
    return accelerations

def rk4_step(bodies, dt):
    """
    Performs one integration step using the Runge-Kutta 4th Order (RK4) method.
    
    Differential Equation: 
    dy/dt = f(t, y)
    Here, y = [position, velocity]
    
    RK4 Formula:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    y_new = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """
    n = len(bodies)
    
    # Current state
    pos_0 = np.array([b.pos for b in bodies])
    vel_0 = np.array([b.vel for b in bodies])
    
    # --- k1 Step ---
    # Velocity at start is vel_0
    # Acceleration at start is computed based on pos_0
    acc_0 = compute_accelerations(bodies, pos_0)
    k1_vel = acc_0
    k1_pos = vel_0

    # --- k2 Step ---
    # Evaluate at t + dt/2, using k1 to estimate state
    pos_1 = pos_0 + 0.5 * dt * k1_pos
    # We don't strictly need updated velocities for acceleration calculation 
    # unless acceleration depended on velocity (drag), but strictly for RK4:
    # vel_1 = vel_0 + 0.5 * dt * k1_vel 
    acc_1 = compute_accelerations(bodies, pos_1)
    k2_vel = acc_1
    k2_pos = vel_0 + 0.5 * dt * k1_vel

    # --- k3 Step ---
    # Evaluate at t + dt/2, using k2
    pos_2 = pos_0 + 0.5 * dt * k2_pos
    acc_2 = compute_accelerations(bodies, pos_2)
    k3_vel = acc_2
    k3_pos = vel_0 + 0.5 * dt * k2_vel

    # --- k4 Step ---
    # Evaluate at t + dt, using k3
    pos_3 = pos_0 + dt * k3_pos
    acc_3 = compute_accelerations(bodies, pos_3)
    k4_vel = acc_3
    k4_pos = vel_0 + dt * k3_vel

    # --- Combine (Weighted Average) ---
    # Update positions and velocities
    new_pos = pos_0 + (dt / 6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
    new_vel = vel_0 + (dt / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

    # Apply updates to body objects
    for i, body in enumerate(bodies):
        body.pos = new_pos[i]
        body.vel = new_vel[i]
        # Append new position to history for trail drawing
        # We only append every X steps to save memory in real apps, but here every step is fine for short sims
        body.trajectory = np.vstack([body.trajectory, new_pos[i]])

# --- Initialization ---
# Masses relative to Sun. Distances in AU. Velocities in AU/Year.
# V_circular approx = 2 * pi / sqrt(r) in these units (since G=4pi^2, M=1)
bodies = [
    Body("Sun",     1.0,         [0, 0, 0],       [0, 0, 0],         'yellow'),
    Body("Mercury", 1.66e-7,     [0.39, 0, 0],    [0, 10.0, 0],      'gray'),  # v approx 1.59 * 2pi
    Body("Venus",   2.45e-6,     [0.72, 0, 0],    [0, 7.38, 0],      'orange'),
    Body("Earth",   3.00e-6,     [1.00, 0, 0],    [0, 6.28, 0],      'blue'),  # v = 2*pi approx
    Body("Mars",    3.23e-7,     [1.52, 0, 0],    [0, 5.08, 0],      'red'),
    Body("Jupiter", 9.54e-4,     [5.20, 0, 0],    [0, 2.75, 0],      'brown')
]

# Simulation parameters
dt = 0.01  # Time step in years (approx 3.65 days)
years = 2  # Total simulation time (run for 12 years to see Jupiter orbit once)
steps = int(years / dt)

# --- Visualization Setup ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set black background for space theme
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.grid(False) 
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False

# Remove axes ticks for cleaner look
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

# Set limits
limit = 6 # AU (Jupiter is at 5.2)
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_zlim(-limit, limit)

# Text annotation for time
time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white')

# Initialize plot elements
lines = []
points = []

for body in bodies:
    # Trajectory line
    line, = ax.plot([], [], [], color=body.color, lw=1, alpha=0.5)
    lines.append(line)
    # Current position marker
    point, = ax.plot([], [], [], marker='o', color=body.color, markersize=6 if body.name != 'Sun' else 10)
    points.append(point)

def init():
    for line, point in zip(lines, points):
        line.set_data([], [])
        line.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
    time_text.set_text('')
    return lines + points + [time_text]

def update(frame):
    # Perform physics step
    rk4_step(bodies, dt)
    
    # Update visual elements
    for i, body in enumerate(bodies):
        # Update Trail (Show last 100 points to avoid clutter, or all for full orbit)
        traj = body.trajectory[-100:] 
        lines[i].set_data(traj[:, 0], traj[:, 1])
        lines[i].set_3d_properties(traj[:, 2])
        
        # Update Marker
        points[i].set_data([body.pos[0]], [body.pos[1]])
        points[i].set_3d_properties([body.pos[2]])
    
    current_time = frame * dt
    time_text.set_text(f'Time: {current_time:.2f} Years')
    
    return lines + points + [time_text]

# Create animation
# Interval is in milliseconds. 
ani = animation.FuncAnimation(fig, update, frames=steps, init_func=init, blit=False, interval=20)

plt.title("N-Body Solar System (RK4 Integration)", color='white')
plt.show()