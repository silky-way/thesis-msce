use bevy::prelude::Component;
use serde::{Deserialize, Serialize};


fn length() -> f32 { 100.0 }
fn width() -> f32 { 3.0 }


#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Lanes {
    #[default]
    Single,
    Double
}

#[derive(Component, Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Road {
    #[serde(default)]
    pub lanes: Lanes,

    #[serde(default = "width")]
    pub width: f32,

    #[serde(default = "length")]
    pub length: f32,
}

impl Default for Road {
    fn default() -> Self {
        Self { lanes: Lanes::default(), width: width(), length: length() }
    }
}