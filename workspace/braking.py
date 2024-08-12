# %% Initialise and graphs per ID seperately

import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

df = pd.read_csv("logs/20_80_complex.csv")

df["velocity"] = (df["velocity_x"]**2 + df["velocity_y"]**2)**0.5
df["acceleration"] = (df["acceleration_x"]**2 + df["acceleration_y"]**2)**0.5

#%% BRAKE GRAPHS

brake = df[df["acceleration_x"] < -0.3]

px.histogram(brake, x="velocity", color="id", nbins=10, labels=dict(value="velocity")).show()
px.histogram(df, x="velocity", color="id", nbins=10).show()


import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Group by vehicle type
vehicle_groups = df.groupby("vehicle")

# Create subplots, one for each vehicle type
fig = make_subplots(rows=len(vehicle_groups), cols=1, 
                    subplot_titles=[f"{vehicle} Speed Distribution" for vehicle in vehicle_groups.groups.keys()],
                    vertical_spacing=0.1)

# Add a histogram for each vehicle type
for i, (vehicle, group) in enumerate(vehicle_groups, start=1):
    fig.add_trace(
        go.Histogram(x=group["velocity"], name=vehicle, nbinsx=20), row=i, col=1
    )
    fig.update_xaxes(title_text="Velocity", row=i, col=1)
    fig.update_yaxes(title_text="Count", row=i, col=1)

# Update layout
fig.update_layout(height=300*len(vehicle_groups), width=800, 
                  title_text="Speed Distribution by Vehicle Type")

fig.show()

# Distribution of braking events by vehicle type
brake = df[df["acceleration_x"] < -0.3]
brake_groups = brake.groupby("vehicle")

fig_brake = make_subplots(rows=len(brake_groups), cols=1, 
                          subplot_titles=[f"{vehicle} Braking Speed Distribution" for vehicle in brake_groups.groups.keys()],
                          vertical_spacing=0.1)

for i, (vehicle, group) in enumerate(brake_groups, start=1):
    fig_brake.add_trace(
        go.Histogram(x=group["velocity"], name=vehicle, nbinsx=20), row=i, col=1
    )
    fig_brake.update_xaxes(title_text="Velocity", row=i, col=1)
    fig_brake.update_yaxes(title_text="Count", row=i, col=1)

fig_brake.update_layout(height=300*len(brake_groups), width=800, 
                        title_text="Braking Speed Distribution by Vehicle Type")

fig_brake.show()
# %%  percentage of heavy braking
brake = df[df["acceleration_x"] < 0.0]
brake_groups = brake.groupby("vehicle")

heavy_brake = df[df["acceleration_x"] < -0.3]
heavy_brake_groups = heavy_brake.groupby("vehicle")

group_labels = ['vehicles']

fig = ff.create_distplot(brake, group_labels)


# %%

import plotly.figure_factory as ff
import numpy as np

# Get the brake data grouped by vehicle
# brake = df[df["acceleration_x"] < 0.0]
# brake_groups = brake.groupby("vehicle")

heavy_brake = df[df["acceleration_x"] < -0.3]
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
    title="Distribution of Velocity for Different Vehicle Types",
    xaxis_title="Velocity (m/s)",
    yaxis_title="Density"
)
fig.show()

# %%
