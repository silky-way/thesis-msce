# %% LOAD EVERYTHING & INITIALISE

import pandas as pd
import plotly.express as px

# Read the CSV file
df = pd.read_csv("logs/20_80.csv")

# %% LINE PLOT OF Y-POSITION
# Sort the dataframe by id and step to ensure correct ordering
df = df.sort_values(['id', 'step'])

# Define the desired y-coordinate and allowed variability
desired_y = -1.5
variability = 0.05
middle = 0

# Create the main scatter plot
fig = px.line(df, x='position_x', y='position_y', color='id',
              title='Vehicle Y-Coordinates Along X-Axis',
              labels={'position_x': 'X Position (meters)', 'position_y': 'Y Position', 'id': 'Vehicle ID'})

# Add lines for the desired y-coordinate and variability range
#fig.add_hline(y=middle, line_dash ="line", line_color="black", annotation_text="Middle of road", annotation_position="top right")
fig.add_hline(y=desired_y, line_dash="solid", line_color="red", annotation_text="Desired y", annotation_position="top right")
fig.add_hline(y=desired_y - variability, line_dash="dot", line_color="green", annotation_text="Lower bound", annotation_position="bottom right")
fig.add_hline(y=desired_y + variability, line_dash="dot", line_color="green", annotation_text="Upper bound", annotation_position="top right")

# Update layout
fig.update_layout(
    legend_title='Vehicles', width=1200, height=600
)

# Display the plot
fig.show()

# %% HISTOGRAM PER ID
# Filter the dataframe for vehicle ID 3
df_vehicle_3 = df[df['id'] == 3]

# Create the histogram
fig = px.histogram(df_vehicle_3, x='position_y', 
                   nbins=int((df_vehicle_3['position_y'].max() - df_vehicle_3['position_y'].min()) / 0.1),
                   labels={'position_y': 'Y Position (meters)', 'count': 'Number of Steps'},
                   title='Distribution of Y Positions for Vehicle ID 3')

# Update layout
fig.update_layout(
    xaxis_title='Y Position (meters)', yaxis_title='Number of Steps',
    bargap=0.1,  # Gap between bars
    width=800, height=500
)

# Add a vertical line for the desired y-coordinate
desired_y = -1.5
fig.add_vline(x=desired_y, line_dash="solid", line_color="red", 
              annotation_text="Desired y", annotation_position="top right")

# Display the plot
fig.show()

# %% DO THIS FOR ALL IDs
# filter data frame
vehicle_ids = sorted(df['id'].unique())

# Create the main figure
fig = make_subplots(rows=1, cols=1, subplot_titles=["Distribution of Y Positions"])

# Create histogram traces for each vehicle ID
traces = []
for vehicle_id in vehicle_ids:
    df_vehicle = df[df['id'] == vehicle_id]
    trace = go.Histogram(
        x=df_vehicle['position_y'],
        nbinsx=int((df_vehicle['position_y'].max() - df_vehicle['position_y'].min()) / 0.1),
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
              {"title": f"Distribution of Y Positions for Vehicle ID {vehicle_id}"}],
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
    xaxis_title='Y Position (meters)', yaxis_title='Number of Steps',
    bargap=0.1, width=800, height=600
)

# Add a vertical line for the desired y-coordinate
desired_y = -1.5
fig.add_vline(x=desired_y, line_dash="solid", line_color="red", 
              annotation_text="Desired y", annotation_position="top right")

# Show the figure
fig.show()


# %% ALL IDs plus BASED ON VEHICLE TYPE
# Calculate deviation from preferred y-position
preferred_y = -1.5

# Get unique vehicle types
vehicle_types = sorted(df['vehicle'].unique())

# Function to create normalized histogram data
def get_histogram(df, vehicle_type):
    vehicle_data = df[df['vehicle'] == vehicle_type]
    hist, bin_edges = np.histogram(vehicle_data['position_y'], bins=np.arange(vehicle_data['position_y'].min(), vehicle_data['position_y'].max() + 0.5, 0.5))
    vehicle_count = vehicle_data['id'].nunique()
    normalized_hist = hist / vehicle_count
    return bin_edges[:-1], normalized_hist
    return bin_edges[:-1], hist


# Create the base figure
initial_vehicle = vehicle_types[0]
x, y = get_histogram(df, initial_vehicle)
fig = px.bar(x=x, y=y, labels={'x': 'Y-Position ', 'y': 'Normalized Count'},
             title=f'Normalized Y-Position Distribution for {initial_vehicle}')

# Add a vertical line at x=0 (preferred position)
fig.add_vline(x=preferred_y, line_width=2, line_dash="solid", line_color="red")

# Update layout for styling
fig.update_layout(
    bargap=0.1, xaxis_title='Y-Position Deviation', yaxis_title='Count (per vehicle)', transition_duration=500
)

# Create sliders
sliders = [
    dict(
        active=0, currentvalue={"prefix": "Vehicle Type: "}, pad={"t": 50},
        steps=[
            dict(
                method='update',
                args=[{'x': [get_histogram(df, vtype)[0]],
                       'y': [get_histogram(df, vtype)[1]]},
                      {'title': f'Y-Position Distribution for {vtype}'}],
                label=str(vtype)
            ) for vtype in vehicle_types
        ]
    )
]

# Update layout to include sliders
fig.update_layout(sliders=sliders)

# Show the figure
fig.show()
# %% INTEGRAL -- VALUE

# df
# df.groupby("vehicle")["position_y"].sum() / df.groupby()


grouped = df.groupby("id")
mapped = grouped["position_y"].sum() / (grouped["step"].max() - grouped["step"].min())

v = pd.DataFrame(grouped["vehicle"].first())
v["value"] = mapped.values
v.groupby("vehicle")["value"].mean()
# %%
