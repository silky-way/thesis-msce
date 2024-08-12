# %% Analysing Bicycle Blind Spots

import numpy as np
import pandas as pd

df = pd.read_csv("logs/20_80.csv")

def calculate_blind_spot(group):
    x = np.stack([group["position_x"].values, group["position_y"].values], axis=1)
    
    # Calculate relative positions and angles
    diffs = x[:, None, :] - x[None, :, :]
    radius = np.linalg.norm(diffs, axis=2)
    angle = np.arctan2(diffs[:, :, 1], diffs[:, :, 0])

    # Determine the blind spot angles
    blind_spot_angles = ((angle > 5 * np.pi / 6) & (angle < 5 * np.pi / 3)) | ((angle > 7 * np.pi / 6) & (angle < 5 * np.pi / 3))

    dead_angle = blind_spot_angles
    dead_dist = (radius < 5) & (radius != 0)

    group["dead"] = (dead_angle * dead_dist).any(axis=1)
    # group["dead"] = (dead_dist).any(axis=0)
    return group

# filtered = df[(df["step"] > 9000) & (df["step"] < 10000)]
# new = filtered.groupby("step").apply(calculate_blind_spot)
# new.groupby("id")["dead"].mean()

new = df.groupby("step").apply(calculate_blind_spot)
# new.groupby("id")["dead"].mean()

# grouped = new.groupby("id")
# v = pd.DataFrame(grouped["id"].first())
# v["dead"] = new.values
# v.groupby("vehicle")["value"].mean()

grouped = new.groupby("id")["dead"].sum()

# %%
q = df[["id", "vehicle"]].drop_duplicates(ignore_index=True)
new = pd.merge(grouped, q, on="id").groupby("vehicle").sum()
new

# %%
