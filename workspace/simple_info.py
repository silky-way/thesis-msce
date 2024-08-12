# %%
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from plotly.subplots import make_subplots
from scipy.ndimage import label

# %% Load trajectory data
def load_trajectory_data(file_path):
    """Load trajectory data from CSV file"""
    df = pd.read_csv(file_path)
    return df

# %% extract information per vehicle
# prep work to use data

df = pd.read_csv("logs/1.csv")
df_1 = df[df.id == 1]
df_2 = df[df.id == 2]
df_3 = df[df.id == 3]

dfs = [df_1, df_2, df_3]

# %% simple insights per vehicle
#  (vel, acc, x & y compared with position)

#for df in dfs:
 #   fig = make_subplots(rows=4, cols=1)

 #   fig.append_trace(go.Scatter(df, x='position_x', y='velocity_x', color='step'), row=1, col=1)
 #   fig.append_trace(go.Scatter(df, x='position_x', y='velocity_y', color='step'), row=2, col=1)
 #   fig.append_trace(go.Scatter(df, x='position_x', y='acceleration_x', color='step'), row=3, col=1)
 #   fig.append_trace(go.Scatter(df, x='position_x', y='acceleration_y', color='step'), row=4, col=1)

#    fig.update_layout(height=800, width=600, title_text="Insights")
#    fig.show()

# %% conflict moments

px.scatter(df_1, y = 'max_interaction_force', x = 'step', color='id')

# %

# px.line(df, y='acceleration_x', x='step', color='id')
# px.line(df, y='velocity_x', x='step', color='id')
# px.scatter(df_1, x='velocity_x', y='acceleration_x', color='step')
#px.scatter(df_2, x='position_y', y='velocity_y', color='step')

# scatterplot
# px.scatter()



# %% Analysing Bicycle Traffic Density

def load_trajectory_data(file_path):
    """Load trajectory data from CSV file"""
    return pd.read_csv(file_path)

def create_grid(path_length, path_width, grid_size):
    """Create a grid over the bike path"""
    nx = int(path_length / grid_size)
    ny = int(path_width / grid_size)
    return np.zeros((ny, nx))

def populate_grid(grid, df, step, grid_size):
    """Populate the grid with bicycle positions for a specific time step"""
    df_step = df[df['step'] == step]
    for _, row in df_step.iterrows():
        x, y = row['position_x'], row['position_y']
        i = int(y / grid_size)
        j = int(x / grid_size)
        if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]:
            grid[i, j] += 1
    return grid

def apply_gaussian_smoothing(grid, sigma):
    """Apply Gaussian smoothing to the grid"""
    return gaussian_filter(grid, sigma=sigma)

def analyze_bicycle_density(df, path_length, path_width, grid_size=1.0, sigma=2.0):
    """Analyze bicycle density for all time steps"""
    steps = df['step'].unique()
    density_maps = []
    
    for step in steps:
        grid = create_grid(path_length, path_width, grid_size)
        populated_grid = populate_grid(grid, df, step, grid_size)
        smoothed_grid = apply_gaussian_smoothing(populated_grid, sigma)
        density_maps.append(smoothed_grid)
    
    return density_maps, steps

def create_interactive_heatmap(df, path_length, path_width, grid_size=1.0, sigma=2.0):
    """Create an interactive heatmap with a slider for time step using Plotly Express"""
    density_maps, steps = analyze_bicycle_density(df, path_length, path_width, grid_size, sigma)
    
    # Create a 3D array of density maps
    density_array = np.array(density_maps)
    
    # Create a grid of x and y coordinates
    ny, nx = density_array.shape[1:]
    x = np.linspace(0, path_length, nx)
    y = np.linspace(0, path_width, ny)
    
    # Create the heatmap
    fig = px.imshow(density_array,
                    aspect='auto',
                    animation_frame=0,  # Use the first axis (time) for animation
                    zmin=0, zmax=density_array.max(),
                    color_continuous_scale='viridis',
                    origin='lower',
                    labels={'animation_frame': 'Time Step'},
                    x=x, y=y,
                    )
    
    # Update layout
    fig.update_layout(
        title='Bicycle Density Heatmap',
        xaxis_title='X position (meters)',
        yaxis_title='Y position (meters)',
    )
    
    # Update colorbar
    fig.update_coloraxes(colorbar_title_text='Density')
    
    # Update slider
    fig.layout.sliders[0].currentvalue.prefix = "Time Step: "
    
    return fig

# %% Analysing Bicycle Blind Spots

def calculate_orientation(velocity):
    """Calculate orientation vector from velocity"""
    magnitude = np.linalg.norm(velocity)
    return velocity / magnitude if magnitude != 0 else np.array([0, 0])

def distance(pos1, pos2):
    """Calculate Euclidean distance between two points"""
    return np.linalg.norm(pos1 - pos2)

def angle_between(v1, v2):
    """Calculate angle between two vectors"""
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def nearness_factor(dist, max_dist):
    """Calculate nearness factor based on distance"""
    return 1 - min(dist / max_dist, 1)

def visibility_factor(angle, max_angle):
    """Calculate visibility factor based on angle"""
    return 1 - min(angle / max_angle, 1)

def bdrd_function(x, beta):
    """Bounded Rational Decision (BDRD) function"""
    return 1 / (1 + np.exp(-beta * x))

def is_in_blind_spot(observer, target, max_angle, max_dist, beta):
    """Determine if the target rider is in the observer's blind spot"""
    rel_pos = target['position'] - observer['position']
    dist = np.linalg.norm(rel_pos)
    
    if dist > max_dist:
        return False
    
    angle = angle_between(observer['orientation'], rel_pos)
    
    vis_factor = visibility_factor(angle, max_angle)
    near_factor = nearness_factor(dist, max_dist)
    
    combined_factor = vis_factor * near_factor
    
    blind_spot_probability = bdrd_function(combined_factor, beta)
    
    return np.random.random() < blind_spot_probability

def analyze_blind_spots(df, max_angle, max_dist, beta):
    """Analyze trajectories to count blind spot occurrences"""
    blind_spot_counts = {rider_id: 0 for rider_id in df['id'].unique()}
    total_steps = df['step'].max() + 1
    
    for step in range(total_steps):
        step_data = df[df['step'] == step]
        
        if step_data.empty:
            continue
        
        positions = step_data[['position_x', 'position_y']].values
        velocities = step_data[['velocity_x', 'velocity_y']].values
        
        for i, observer_row in step_data.iterrows():
            observer = {
                'id': observer_row['id'],
                'position': np.array([observer_row['position_x'], observer_row['position_y']]),
                'velocity': np.array([observer_row['velocity_x'], observer_row['velocity_y']]),
                'orientation': calculate_orientation(np.array([observer_row['velocity_x'], observer_row['velocity_y']]))
            }
            
            for j, target_row in step_data.iterrows():
                if observer['id'] != target_row['id']:
                    target = {
                        'id': target_row['id'],
                        'position': np.array([target_row['position_x'], target_row['position_y']])
                    }
                    
                    if is_in_blind_spot(observer, target, max_angle, max_dist, beta):
                        blind_spot_counts[observer['id']] += 1
    
    # Convert counts to frequencies
    total_time = total_steps * 60  # Assuming 1 second time step, 60 sec per minute
    blind_spot_frequencies = {rider: count / total_time for rider, count in blind_spot_counts.items()}
    
    return blind_spot_frequencies

#%% Analyzing Bicycle Braking Behaviour

import pandas as pd
import plotly.express as px

df = pd.read_csv("logs/2.csv")

df["velocity"] = (df["velocity_x"]**2 + df["velocity_y"]**2)**0.5
df["acceleration"] = (df["acceleration_x"]**2 + df["acceleration_y"]**2)**0.5

brake = df[df["acceleration_x"] < -0.3]

px.histogram(brake, x="velocity", color="id", nbins=10, labels=dict(value="velocity")).show()
px.histogram(df, x="velocity", color="id", nbins=10).show()

# %% Analyzing Bicycle Density through convolution

import pandas as pd
import plotly.express as px
import numpy as np
from scipy.ndimage import gaussian_filter

df = pd.read_csv("logs/1.csv")

grid = np.zeros((1000, 24))

pos_x = df[df["step"] == 800]["position_x"].tolist()
pos_y = df[df["step"] == 800]["position_y"].tolist()

for x, y in zip(pos_x, pos_y):
    grid[np.clip(int(x * 10), 0, 1000), np.clip(int((y + 2) * 6), 0, 24)] = 1

k = 5
grid = gaussian_filter(grid, sigma=[10*k, k])
grid = grid ** 4

px.imshow(grid.T, aspect='auto').show()

# %% Main execution

if __name__ == "__main__":
    # Load data from CSV: don't forget to adapt file path!
    file_path = "logs/2.csv"
    df = load_trajectory_data(file_path)

    # Set parameters
    path_length = 100  # meters, adjust based on your simulation
    path_width = 3  # meters, adjust based on your simulation
    grid_size = 1.0  # meters
    sigma = 2.0
    max_angle = np.pi / 3  # 60 degrees
    max_dist = 10  # meters
    beta = 5  # BDRD function parameter

     # Create and display the interactive heatmap
    fig = create_interactive_heatmap(df, path_length, path_width, grid_size, sigma)
    fig.show()

     # Analyze blind spots
    blind_spot_frequencies = analyze_blind_spots(df, max_angle, max_dist, beta)
    for rider, frequency in blind_spot_frequencies.items():
        print(f"Rider {rider} was in someone's blind spot {frequency:.2f} sec per min on average")

    # Analyze braking
    braking_events, overall_braking_stats, overall_force_stats = analyze_braking(df)
    print(f"Braking {overall_braking_stats}")
    print(f"Force {overall_force_stats}")
    plot_braking_events('1', df, braking_events)



# %%