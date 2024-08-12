# %%
import pandas as pd
import plotly.express as px
import numpy as np
from scipy.ndimage import gaussian_filter

df = pd.read_csv("logs/scaleup.csv")

# %% CONVOLUTION FOR CERTAIN TIME STEP

grid = np.zeros((1000, 24))

pos_x = df[df["step"] == 200]["position_x"].tolist()
pos_y = df[df["step"] == 200]["position_y"].tolist()

for x, y in zip(pos_x, pos_y):
    grid[np.clip(int(x * 10), 0, 1000), np.clip(int((y + 2) * 6), 0, 24)] = 1

k = 5
grid = gaussian_filter(grid, sigma=[10*k, k])
grid = grid ** 4

px.imshow(grid.T, aspect='auto').show()

# %%

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.ndimage import gaussian_filter

# Load the data
df = pd.read_csv("logs/20_80.csv")

# Get the unique time steps
time_steps = sorted(df['step'].unique())

# Create a function to generate the grid for a given time step
def create_grid(step):
    grid = np.zeros((1000, 24))
    pos_x = df[df["step"] == step]["position_x"].tolist()
    pos_y = df[df["step"] == step]["position_y"].tolist()
    
    for x, y in zip(pos_x, pos_y):
        grid[np.clip(int(x * 10), 0, 999), np.clip(int((y + 2) * 6), 0, 23)] = 1
    
    k = 5
    grid = gaussian_filter(grid, sigma=[10*k, k])
    grid = grid ** 4
    
    return grid

# Create the initial grid
initial_grid = create_grid(time_steps[0])

# Create the figure
fig = make_subplots(rows=1, cols=1)

# Add the heatmap
heatmap = go.Heatmap(z=initial_grid.T, colorscale='Viridis')
fig.add_trace(heatmap)

# Update layout
fig.update_layout(
    title="Convolution over Time",
    xaxis_title="X Position",
    yaxis_title="Y Position",
)

# Add slider
steps = []
for step in time_steps:
    step = dict(
        method="update", args=[{"z": [create_grid(step).T]}], label=str(step)
    )
    steps.append(step)

sliders = [dict(
    active=0, currentvalue={"prefix": "Time Step: "}, pad={"t": 50}, steps=steps
)]

fig.update_layout(
    sliders=sliders
)

# Show the figure
fig.show()

# %% MEAN MAX DENSITY

from scipy.stats import gaussian_kde

# Function to calculate density
def calculate_density(group):
    positions = group[['position_x', 'position_y']].values
    kde = gaussian_kde(positions.T)
    return kde(positions.T).max()

# Calculate max density per timestep
max_densities = df.groupby('step').apply(calculate_density)

# Calculate mean of max densities
mean_max_density = max_densities.mean()

print(f"Mean of maximum densities: {mean_max_density}")

# %%
