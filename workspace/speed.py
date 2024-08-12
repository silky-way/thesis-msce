# %% LOAD EVERYTHING & INITIALISE
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Read the CSV file
df = pd.read_csv("logs/7.csv")
df["velocity"] = (df["velocity_x"]**2 + df["velocity_y"]**2)**0.5

# %% GRAPH PER VEHICLE
# filter data frame
vehicle_ids = sorted(df['id'].unique())

# Create the main figure
fig = make_subplots(rows=1, cols=1, subplot_titles=["Distribution of Speed"])

# Create histogram traces for each vehicle ID
traces = []
for vehicle_id in vehicle_ids:
    df_vehicle = df[df['id'] == vehicle_id]
    trace = go.Histogram(
        x=df_vehicle['velocity'],
        name=f'Vehicle {vehicle_id}',
        visible=False
    )
    traces.append(trace)

# Make the first trace visible
traces[0].visible = True

# Add all traces to the figure
for trace in traces:
    fig.add_trace(trace)

# Create slider steps
steps = []
for i, vehicle_id in enumerate(vehicle_ids):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(traces)},
              {"title": f"Distribution of Speed for Vehicle ID {vehicle_id}"}],
        label=str(vehicle_id)
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

# Create slider
sliders = [dict(
    active=0,
    currentvalue={"prefix": "Vehicle ID: "}, pad={"t": 50}, steps=steps
)]

# Update layout
fig.update_layout(
    sliders=sliders,
    xaxis_title='Speed (meters per sec)', yaxis_title='Number of Steps',
    bargap=0.1, width=800, height=600
)

# Show the figure
fig.show()

# %% MEAN PER VEHICLE TYPE
# Calculate speed for all rows
df['speed'] = np.sqrt(df['velocity_x']**2 + df['velocity_y']**2)

# Get unique vehicle types
vehicle_types = sorted(df['vehicle'].unique())

# Function to create normalized histogram data
def get_normalized_histogram(df, vehicle_type):
    vehicle_data = df[df['vehicle'] == vehicle_type]
    hist, bin_edges = np.histogram(vehicle_data['speed'], bins=np.arange(0, vehicle_data['speed'].max() + 0.5, 0.5))
    vehicle_count = vehicle_data['id'].nunique()
    normalized_hist = hist / vehicle_count
    return bin_edges[:-1], normalized_hist

# Create the base figure
initial_vehicle = vehicle_types[0]
x, y = get_normalized_histogram(df, initial_vehicle)
fig = px.bar(x=x, y=y, labels={'x': 'Speed', 'y': 'Normalized Count'},
             title=f'Normalized Speed Distribution for {initial_vehicle}')

# Update layout for styling
fig.update_layout(
    bargap=0.1, xaxis_title='Speed', yaxis_title='Normalized Count (per vehicle)',
    transition_duration=500
)

# Create sliders
sliders = [
    dict(
        active=0,
        currentvalue={"prefix": "Vehicle Type: "},
        pad={"t": 50},
        steps=[
            dict(
                method='update',
                args=[{'x': [get_normalized_histogram(df, vtype)[0]], 'y': [get_normalized_histogram(df, vtype)[1]]},
                      {'title': f'Normalized Speed Distribution for {vtype}'}],
                label=str(vtype)
            ) for vtype in vehicle_types
        ]
    )
]

# Update layout to include sliders
fig.update_layout(sliders=sliders)

# Show the figure

# %% VALUES TO COMPARE


