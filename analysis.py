# %% LOAD EVERYTHING & INITIALISE
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
import plotly.figure_factory as ff


# Read the CSV file
df = pd.read_csv("logs/0_100_complex.csv")

# Adapt velocity
df["velocity"] = (df["velocity_x"]**2 + df["velocity_y"]**2)**0.5
df["acceleration"] = (df["acceleration_x"]**2 + df["acceleration_y"]**2)**0.5

# Get the data grouped by vehicle
groups = df.groupby("vehicle")

# %% SPEED: DISTPLOT with velocity NORM density per vehicle TYPE

# Extract the acceleration data for each vehicle type
hist_data = []
group_labels = []
for name, group in groups:
    hist_data.append(group["velocity"])
    group_labels.append(name)

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(
    title="Distribution of Velocity for Different Vehicle Types 0/100",
    xaxis_title="Velocity (m/s)",
    yaxis_title="Density"
)
fig.show()


# %% DEVIATION: DISTPLOT with MEAN per vehicle TYPE
# Calculate deviation from preferred y-position mean per vehicle type

preferred_y = -1.5

# Extract the acceleration data for each vehicle type
hist_data = []
group_labels = []
for name, group in groups:
    hist_data.append(group["position_y"])
    group_labels.append(name)

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(
    title="Distribution of Perpendicular Deviation for Different Vehicle Types 50/50",
    xaxis_title="Y-Position Deviation (m)",
    yaxis_title="Density"
)
fig.add_vline(x=preferred_y, line_width=2, line_dash="solid", line_color="red")
fig.show()

 # %% BRAKING: DISTPLOT of braking per vehicle TYPE
# Distribution of braking events' velocity by vehicle type
# Get the brake data grouped by vehicle
brake = df[df["acceleration_x"] < 0.0]
brake_groups = brake.groupby("vehicle")

heavy_brake = df[df["acceleration_x"] < -0.2]
heavy_brake_groups = heavy_brake.groupby("vehicle")

# Extract the acceleration data for each vehicle type
hist_data = []
group_labels = []
for name, group in heavy_brake_groups:
    hist_data.append(group["velocity"])
    group_labels.append(name)

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.update_layout(
    title="Distribution of Velocities for Different Vehicle Types with Heavy Braking",
    xaxis_title="Velocity (m/s)",
    yaxis_title="Density"
)
fig.show()

# %% DENSITY: VALUE MEAN MAX of density per time STEP with GAUSSIAN
# Value for mean of max density per time step

def create_grid(group):
    grid = np.zeros((1000, 24))
    pos_x = group["position_x"].tolist()
    pos_y = group["position_y"].tolist()
    
    for x, y in zip(pos_x, pos_y):
        grid[np.clip(int(x * 10), 0, 999), np.clip(int((y + 2) * 6), 0, 23)] = 1
    
    k = 5
    grid = gaussian_filter(grid, sigma=[10*k, k])
    grid = (grid * 1e3) ** 4
    
    return grid

# Function to calculate density
def calculate_density(group):
    return create_grid(group).flatten().max()

# Calculate max density per timestep
max_densities = df.groupby('step').apply(calculate_density)

# Calculate mean of max densities
mean_max_density = max_densities.mean()

print(f"Mean of maximum densities: {mean_max_density}")

# %% DEVIATION: VALUE with mean of sum (INTEGRAL)

grouped = df.groupby("id")

# check per id for which steps the sum needs to be taken
mapped = grouped["position_y"].sum() / (grouped["step"].max() - grouped["step"].min())

# aggregate over vehicle type
v = pd.DataFrame(grouped["vehicle"].first())
v["value"] = mapped.values

# take mean of integral per vehicle type
v.groupby("vehicle")["value"].mean()

# %% SPEED: VALUE mean per vehicle TYPE  

print(df.groupby("vehicle")["velocity"].mean())

#%% BRAKING: VALUE mean per vehicle TYPE

df["decelerate"] = df["acceleration_x"] < 0.0
df["braking"] = df["acceleration_x"] < -0.2

print('- Deceleration %', df.groupby("vehicle")["decelerate"].mean())
print('- Heavy braking %', df.groupby("vehicle")["braking"].mean())
print('- Heavy braking of deceleration %', df.groupby("vehicle")["braking"].mean() *100 / df.groupby("vehicle")["decelerate"].mean())

# %% BLIND SPOT: VALUE with mean per vehicle TYPE

def calculate_blind_spot(group):
    x = np.stack([group["position_x"].values, group["position_y"].values], axis=1)
    
    # Calculate relative positions and angles (angle in polar coordinates)
    diffs = x[:, None, :] - x[None, :, :]
    radius = np.linalg.norm(diffs, axis=2)
    angle = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])

    # Determine the blind spot angles
    blind_spot_angles = ((angle < 5 * np.pi / 6) & (angle > np.pi / 3))   #| ((angle > 7 * np.pi / 6) & (angle < 5 * np.pi / 3))

    dead_angle = blind_spot_angles
    dead_dist = (radius < 3) & (radius != 0)

    group["dead"] = (dead_angle * dead_dist).any(axis=1)

    return group

new = df.groupby("step").apply(calculate_blind_spot)
grouped = new.groupby("id")["dead"].sum()

q = df[["id", "vehicle"]].drop_duplicates(ignore_index=True)
new = pd.merge(grouped, q, on="id").groupby("vehicle").sum()

print(new)

# %%
