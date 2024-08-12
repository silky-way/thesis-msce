//use std::sync::atomic::{AtomicU64, Ordering};
use bevy::prelude::{Color, Component, Vec2};
use serde::{Serialize, Deserialize};

#[derive(Copy, Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Vehicle {
    #[default]
    Bike,
    Ebike, 
    SpeedPedelec, // in snake_case this becomes speed_pedelec !
    Pedestrian
}


impl Vehicle {
    pub fn size(&self) -> (f32, f32) {
        match self {
            Vehicle::Bike => (2.0, 0.5),
            Vehicle::Ebike => (2.0, 0.5),
            Vehicle::SpeedPedelec => (2.0, 0.8),
            Vehicle::Pedestrian => (1.0, 0.5)
        }
    }

    pub fn color(&self) -> Color {
        match self {
            Vehicle::Bike => Color::LIME_GREEN,
            Vehicle::Ebike => Color::WHITE,
            Vehicle::SpeedPedelec => Color::SALMON,
            Vehicle::Pedestrian => Color::CYAN
        }
    }

    pub fn interaction_radius(self, phi: f32) -> f32 {
        let (length, width) = self.size();
        let inner = (length * length - width * width) * phi.cos() * phi.cos() / (length * length);
        width / (1f32 - inner).sqrt()
    }

    pub fn interaction(target: Vehicle, source: Vehicle) -> (f32, f32) {
        let (a, b) = match (target, source) { 
            // A = interaction strength (m/s^2), B = interaction range (m)
            (Vehicle::Bike, Vehicle::Bike) => (2.0, 5.0),
            (Vehicle::Bike, Vehicle::Ebike) => (2.0, 5.0),
            (Vehicle::Bike, Vehicle::SpeedPedelec) => (2.0, 5.0),
            (Vehicle::Bike, Vehicle::Pedestrian) => (2.0, 5.0),
            (Vehicle::Ebike, Vehicle::Bike) => (2.0, 5.0),
            (Vehicle::Ebike, Vehicle::Ebike) => (2.0, 5.0),
            (Vehicle::Ebike, Vehicle::SpeedPedelec) => (2.0, 5.0),
            (Vehicle::Ebike, Vehicle::Pedestrian) => (2.0, 5.0),
            (Vehicle::SpeedPedelec, Vehicle::Bike) => (3.0, 5.0),
            (Vehicle::SpeedPedelec, Vehicle::Ebike) => (3.0, 5.0),
            (Vehicle::SpeedPedelec, Vehicle::SpeedPedelec) => (3.0, 5.0),
            (Vehicle::SpeedPedelec, Vehicle::Pedestrian) => (3.0, 5.0),
            (Vehicle::Pedestrian, Vehicle::Bike) => (2.0, 5.0),
            (Vehicle::Pedestrian, Vehicle::Ebike) => (2.0, 5.0),
            (Vehicle::Pedestrian, Vehicle::SpeedPedelec) => (2.0, 5.0),
            (Vehicle::Pedestrian, Vehicle::Pedestrian) => (2.0, 5.0),
        };
        (a * 1.0, b * 1.0)
    }

    pub fn form_factor(target: Vehicle) -> f32 {
        match target {
            Vehicle::Bike => 0.2,
            Vehicle::Ebike => 0.1,
            Vehicle::SpeedPedelec => 0.2,
            Vehicle::Pedestrian => 0.2,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tendency {
    None,
    Following,
    Shoulder
}

// https://github.com/serde-rs/serde/issues/368
fn one() -> f32 { 1.0 }
fn five() -> f32 { 5.0 }
fn end() -> Vec2 { 1000.0 * Vec2::X }


#[derive(Component, Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Commuter {
    pub id: u64,

    #[serde(default)]
    pub vehicle: Vehicle,

    #[serde(default)]
    pub start_position: Vec2,

    #[serde(default = "end")]
    pub end_position: Vec2,

    #[serde(default = "five")]
    pub desired_speed: f32, // m/s

    #[serde(default = "five")]
    pub start_speed: f32, // m/s

    #[serde(default = "one")]
    pub relaxation_time: f32,

    #[serde(default)]
    pub spawn_step: u64
}