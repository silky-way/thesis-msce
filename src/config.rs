use std::fs::File;
use std::path::Path;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use bevy::ecs::system::Resource;
use crate::commuter::Commuter;
use crate::road::Road;


#[derive(Debug, Serialize, Deserialize)]
pub struct Simulation {
    #[serde(default)]
    border_force: f32,
}

impl Default for Simulation {
    fn default() -> Self {
        Self { border_force: 1.001 }
    }
}

#[derive(Debug, Serialize, Deserialize, Resource)]
pub struct Scenario {
    pub name: String,

    #[serde(default)]
    pub simulation: Simulation,

    #[serde(default)]
    pub road: Road,

    pub commuters: Vec<Commuter>,
}

#[derive(Debug, Serialize, Deserialize, Resource)]
pub struct Config {
    pub debug_acceleration_scale: f32,
    pub debug_velocity_scale: f32,

    pub border_force: f32,
    pub border_cap: f32,
    pub border_decay: f32,

    pub ema_factor: f32,
    pub nudge_acceleration_threshold: f32,
    pub nudge_velocity_offset: f32,
    pub nudge_strength: f32,

    pub y_velocity_dampening: f32,

    pub road_centre_offset: f32
}

pub fn parse<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let config: T = serde_json::from_reader(file)?;
    Ok(config)
}