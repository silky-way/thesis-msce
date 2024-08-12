mod config;
mod commuter;
mod road;

use bevy::prelude::*;
use bevy_prototype_lyon::prelude::*;
use bevy::gizmos::gizmos::Gizmos;
use serde::{Serialize, Deserialize};
use bevy::render::camera::ScalingMode;
use crate::commuter::{Commuter, Vehicle};
use crate::road::Road;
use bevy::sprite::MaterialMesh2dBundle;
use bevy::sprite::Mesh2dHandle;
use crate::config::{parse, Config, Scenario};
use csv::Writer;
use std::collections::HashMap;
use std::fs::File;
use bevy::app::AppExit;


#[derive(Component, Clone, Debug, PartialEq, Default)]
struct Verlet {
    previous: Vec2,
    position: Vec2,

    velocity: Vec2,
    acceleration: Vec2,

    interaction_forces: HashMap<u64, Vec2>,
    border_force: Vec2,
    driving_force: Vec2,
    nudge_force: Vec2,
}

#[derive(Component, Copy, Clone, Debug, PartialEq)]
struct Rolling {
    velocity: Vec2,
    acceleration: Vec2,
    nudge: Vec2
}

#[derive(Serialize, Deserialize)]
struct Record {
    id: u64,
    vehicle: Vehicle,
    step: u64,

    position_x: f32,
    position_y: f32,

    velocity_x: f32,
    velocity_y: f32,

    acceleration_x: f32,
    acceleration_y: f32,

    rolling_velocity_x: f32,
    rolling_acceleration_x: f32,

    // serde csv flatten doesn't work, so we have to write it out manually
    max_interaction_force: f32,
    max_interaction_id: Option<u64>,

    border_force: f32,
    nudge_force: f32,

    driving_force_x: f32,
    driving_force_y: f32,
}

impl Verlet {
    pub fn from_start_and_velocity(start: Vec2, velocity: Vec2) -> Self {
        Self { previous: start - velocity, position: start, velocity, ..Default::default() }
    }
}

#[derive(Resource)]
pub struct Step(pub u64);

#[derive(Resource)]
pub struct Despawned(pub usize);

#[derive(Resource)]
pub struct Logger(pub Writer<File>);


// main function that runs the full program
// don't forget to adapt the logger/config/scenario file if needed !
// don't forget to add systems if created !
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let logger = Writer::from_path("logs/0_100_complex.csv").unwrap();

    let config = parse::<Config>("config.json")?;
    let scenario = parse::<Scenario>("scenarios/0_100_complex.json").unwrap();

    App::new()
        .insert_resource(Msaa::Sample4)
        .insert_resource(Logger(logger))
        .insert_resource(Step(0))
        .insert_resource(Despawned(0))
        .insert_resource(config)
        .insert_resource(scenario)

        // Normal version
        // .add_plugins(DefaultPlugins)
        // .add_plugins(ShapePlugin)

        // Headless version
        .add_plugins(MinimalPlugins)
        
        .add_systems(First, (spawn_commuters, despawn_commuters, reset_step))
        .add_systems(Update, (interaction_forces, border_forces, driving_forces, nudge_forces))
        .add_systems(PostUpdate, (integration_step, ))

        // Normal version
        // .add_systems(Last, (translate_step, debug_mode, extract_data))  
        // .add_systems(Startup, (normal_setup, ))

        // Headless version
        .add_systems(Last, (translate_step, extract_data)) 
        .add_systems(Startup, (headless_setup, ))
        
        .run();

    Ok(())
}

// Debugging purposes, visualizing various forces acting on commuters
fn debug_mode(mut gizmos: Gizmos, mut commuters: Query<(&Verlet, &Commuter, &Rolling)>, cfg: Res<Config>) {
    for (verlet, _, rolling) in commuters.iter_mut() {
        let start = verlet.position;

        // Calculate end points for different forces/velocities
        // Each force is scaled by a debug factor for better visualization
        let end_acc = start + verlet.acceleration * cfg.debug_acceleration_scale;
        let end_border = start + verlet.border_force * cfg.debug_acceleration_scale;
        let end_interaction = start + verlet.interaction_forces.values().sum::<Vec2>() * cfg.debug_acceleration_scale;
        let end_nudge = start + rolling.nudge * cfg.debug_acceleration_scale;
        let end_vel = start + verlet.velocity * cfg.debug_velocity_scale;

        // Draw lines for each force/velocity
        gizmos.line_2d(start, end_acc, Color::RED);
        gizmos.line_2d(start, end_vel, Color::BLUE);
        gizmos.line_2d(start, end_border, Color::PINK);
        gizmos.line_2d(start, end_interaction, Color::GREEN);
        gizmos.line_2d(start, end_nudge, Color::GRAY)
    }  
}

// extract data from simulation to usable csv (iterating over all entities with Verlet/commuter/rolling components)
fn extract_data(mut commuters: Query<(&Verlet, &Commuter, &Rolling)>, mut writer: ResMut<Logger>, step: Res<Step>) {
    for (verlet, commuter, rolling) in commuters.iter_mut() {
        
        // Find the maximum interaction force and its associated ID
        let option = verlet.interaction_forces.iter().max_by(|a, b| a.1.length().partial_cmp(&b.1.length()).unwrap());
        let max_id = option.map(|(id, _)| *id);
        let max_force = option.map(|(_, force)| *force).unwrap_or(Vec2::ZERO);

        // Create a record per ID & timestep with various variables
        let record = Record {
            id: commuter.id,
            vehicle: commuter.vehicle,
            step: step.0,
            position_x: verlet.position.x,
            position_y: verlet.position.y,
            velocity_x: verlet.velocity.x * 60.0, // converting to units per sec (60 evaluations per sec)
            velocity_y: verlet.velocity.y * 60.0,
            acceleration_x: verlet.acceleration.x,
            acceleration_y: verlet.acceleration.y,
            rolling_velocity_x: rolling.velocity.x,
            rolling_acceleration_x: rolling.acceleration.x,
            
            max_interaction_force: max_force.length(),
            max_interaction_id: max_id,

            border_force: verlet.border_force.y,
            nudge_force: verlet.nudge_force.y,
            driving_force_x: verlet.driving_force.x,
            driving_force_y: verlet.driving_force.y,
        };

        writer.0.serialize(record).unwrap();
    }
    writer.0.flush().unwrap(); // Ensure all data is written to the output
}

// function to define interaction foces for each vehicle (setup for interaction_forces function)
fn calculate_interaction(s_verlet: &Verlet, s_commuter: &Commuter, t_verlet: &mut Verlet, t_commuter: &Commuter) {
    // Calculate interaction parameters (target = vehicle for whom interaction is calculated)
    let (interaction_strength, interaction_range) = Vehicle::interaction(t_commuter.vehicle,s_commuter.vehicle);
    let lambda = Vehicle::form_factor(t_commuter.vehicle);

    // Calculate distance and normalized direction vector
    let distance_between = s_verlet.position.distance(t_verlet.position);
    let direction_between = (s_verlet.position - t_verlet.position).normalize();

    // calculate the direction to the end point of the target vehicle
    let direction_to_end = (s_commuter.end_position - s_verlet.position).normalize();

    let phi = direction_to_end.angle_between(direction_between);
    let form_factor = lambda + (1.0 - lambda) * (1.0 + phi.cos()) / 2.0;

    // Calculate R_t and R_s
    let target_radius = t_commuter.vehicle.interaction_radius(phi);
    let source_radius = s_commuter.vehicle.interaction_radius(phi);

    // speed check due to interaction force too high after passing by
    let h = s_commuter.desired_speed - t_commuter.desired_speed;
    let x = std::f32::consts::E.powf(-1.20*h);
    let y = 1.0;
    let speed_check: f32 = x.min(y);
    // print!("{}", speed_check);

    let interaction_force = speed_check * interaction_strength * 
        (f32::exp((target_radius + source_radius - distance_between) / interaction_range))
        * direction_between * form_factor;

    t_verlet.interaction_forces.insert(s_commuter.id, -interaction_force);
}

// function to actually calculate interaction forces between vehicles
fn interaction_forces(mut commuters: Query<(&mut Verlet, &Commuter)>) {
    // combining two for loops into one by using cartesian product
    // to get the position and speed of both target and source

    let mut combinations = commuters.iter_combinations_mut();
    while let Some([(mut s_verlet, s_commuter), (mut t_verlet, t_commuter)]) = combinations.fetch_next() {
        calculate_interaction(&s_verlet, s_commuter, &mut t_verlet, t_commuter);
        calculate_interaction(&t_verlet, t_commuter, &mut s_verlet, s_commuter);
    }
}

// force that propels entity to the end of the road/destination
fn driving_forces(mut commuters: Query<(&mut Verlet, &Commuter)>){
    for (mut verlet, commuter) in commuters.iter_mut() {
        let direction_to_end = (commuter.end_position - verlet.position).normalize();
        let driving_force = (commuter.desired_speed * direction_to_end / 60.0 - verlet.velocity) / commuter.relaxation_time;

        verlet.driving_force = driving_force;
    }
}

// force that pushes entities away from the border when coming too close (keep them in designated area)
fn border_forces(mut commuters: Query<(&mut Verlet, &Commuter)>, _roads: Query<&Road>, cfg: Res<Config>) {
    // a) Iterates through all commuters using a query system
    // b) Calculates the distance of each commuter from the center of the area
    // c) Applies a force based on this distance
    for (mut verlet, _commuter) in commuters.iter_mut() {
        let distance_to_centre = verlet.position.y + cfg.road_centre_offset;

        let force = cfg.border_force * distance_to_centre.abs().powf(cfg.border_decay) * distance_to_centre.signum();
        verlet.border_force = Vec2::new(0.0, -force);
    }
}

// easy way to prevent entities of getting stuck/moving too slow
fn nudge_forces(mut commuters: Query<(&mut Verlet, &Commuter, &Rolling)>, cfg: Res<Config>) {
    // a) Iterates through commuters
    // b) Checks if a commuter is stuck based on certain criteria (acc_x below threshold and vel_x below desired speed by a certain threshold)
    // c) Applies a "nudge" force if necessary
    for (mut verlet, commuter, rolling) in commuters.iter_mut() {
        // Some heuristic to detect if the commuter is stuck
        if rolling.acceleration.x < cfg.nudge_acceleration_threshold && rolling.velocity.x < (commuter.desired_speed - cfg.nudge_velocity_offset) / 60.0 {
            verlet.nudge_force = Vec2::new(0.0, cfg.nudge_strength);
        }
    }
}

// reset calculated variables to zero eacht time step
fn reset_step(mut query: Query<&mut Verlet>) {
    for mut verlet in query.iter_mut() {
        verlet.acceleration = Vec2::ZERO;
        verlet.interaction_forces.clear();

        verlet.border_force = Vec2::ZERO;
        verlet.driving_force = Vec2::ZERO;
        verlet.nudge_force = Vec2::ZERO;
    }
}

// uupdate the position using the acceleration of the simulation step
// for each Verlet component
fn integration_step(mut query: Query<(&mut Verlet, &mut Rolling)>, cfg: Res<Config>) {
    for (mut verlet, mut rolling) in query.iter_mut() {
        let dt = 1.0 / 60.0; // typical for 60 FPS simulations

        let interaction_force = verlet.interaction_forces.values().sum::<Vec2>();
        verlet.acceleration = interaction_force + verlet.driving_force + verlet.border_force + rolling.nudge;

        // Update the rolling averages
        rolling.acceleration = rolling.acceleration * cfg.ema_factor + verlet.acceleration * (1.0 - cfg.ema_factor);
        rolling.velocity = rolling.velocity * cfg.ema_factor + verlet.velocity * (1.0 - cfg.ema_factor);
        rolling.nudge = rolling.nudge * cfg.ema_factor + verlet.nudge_force * (1.0 - cfg.ema_factor);

        let mut velocity = verlet.position - verlet.previous;
        velocity.y *= cfg.y_velocity_dampening;

        verlet.velocity = velocity;
        verlet.previous = verlet.position;

        verlet.position.x += velocity.x + verlet.acceleration.x * dt * dt;
        verlet.position.y += velocity.y + verlet.acceleration.y * dt * dt;
    }
}

// translate vehicles to new position
fn translate_step(mut query: Query<(&mut Transform, &Verlet)>) {
    for (mut transform, verlet) in query.iter_mut() {
        transform.translation.x = verlet.position.x;
        transform.translation.y = verlet.position.y;
    }
}

// The signature is slightly long but we needs lots of stuff to check if the simulation is done
fn despawn_commuters(mut commands: Commands, mut query: Query<(Entity, &Verlet, &Commuter)>, road: Query<&Road>, mut despawned: ResMut<Despawned>, scenario: Res<Scenario>, mut exit: EventWriter<AppExit>) {
    let road = road.single();

    for (entity, verlet, commuter) in query.iter_mut() {
        if verlet.position.x > road.length {
            commands.entity(entity).despawn();
            despawned.0 += 1;
            println!("{} reached the end", commuter.id);
        }
    }
    if despawned.0 == scenario.commuters.len() {
        exit.send(AppExit);
    }
}

fn spawn_commuters(mut commands: Commands, scenario: Res<Scenario>, mut step: ResMut<Step>) {
    for commuter in &scenario.commuters {
        if commuter.spawn_step == step.0 {
            commands.spawn(make_commuter_bundle(commuter.clone()));
        }
    }

    step.0 += 1;
}

fn make_commuter_bundle(commuter: Commuter) -> impl Bundle {
    let translation = Vec3::new(0.0, 0.0, commuter.id as f32);

    let velocity = Vec2::new(commuter.start_speed, 0.0) / 60.0 ;
    let verlet = Verlet::from_start_and_velocity(commuter.start_position, velocity);

    let (a, b) = commuter.vehicle.size();
    let color = commuter.vehicle.color();

    let points = [Vec2::new(-a, b), Vec2::new(a, b), Vec2::new(a, -b), Vec2::new(-a, -b)];

    let shape = shapes::RoundedPolygon {
        points: points.into_iter().collect(),
        radius: 10.,
        closed: false,
    };

    (
        ShapeBundle {
            path: GeometryBuilder::build_as(&shape),
            spatial: SpatialBundle{
                transform: Transform::from_translation(translation),
                ..Default::default()
            },
            ..Default::default()
        },
        Fill::color(color),
        verlet,
        commuter,
        Rolling{ velocity: Vec2::ZERO, acceleration: Vec2::ZERO, nudge: Vec2::ZERO },
    )
}


fn headless_setup(mut commands: Commands, scenario: Res<Scenario>) {
    commands.spawn(scenario.road);
}

fn normal_setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut materials: ResMut<Assets<ColorMaterial>>, scenario: Res<Scenario>) {
    
    let (top, bottom) = (scenario.road.width, -scenario.road.width);

    commands.spawn(scenario.road);

    let mut builder = PathBuilder::new();
    builder.move_to(Vec2::new(0.0, top));
    builder.line_to(Vec2::new(100.0, top));

    builder.move_to(Vec2::new(0.0, bottom));
    builder.line_to(Vec2::new(100.0, bottom));
    builder.close();

    commands.spawn((
        ShapeBundle { path: builder.build(), ..default() },
        Stroke::new(Color::BLACK, 0.3)
    ));

    let mut builder = PathBuilder::new();
    for i in (0..100).step_by(2) {
        builder.move_to(Vec2::new(i as f32 + 0.5, 0.0));
        builder.line_to(Vec2::new((i+1) as f32 + 0.5, 0.0));
    }
    builder.close();
    commands.spawn((
        ShapeBundle { path: builder.build(), ..default() },
        Stroke::new(Color::WHITE, 0.15)
    ));

    let color = Color::rgb(0.3, 0.3, 0.3);
    let shape = Mesh2dHandle(meshes.add(Rectangle::new(100.0, 2.0 * scenario.road.width)));
    commands.spawn(MaterialMesh2dBundle {
        mesh: shape,
        material: materials.add(color),
        transform: Transform::from_xyz(50.0, 0.0, -100.0),
        ..default()
    });

    let projection = OrthographicProjection {
        near: -1000.0,
        scaling_mode: ScalingMode::WindowSize(10.0),
        ..Default::default()
    };

    commands.spawn(Camera2dBundle{
        projection,
        transform: Transform::from_xyz(50.0, 0.0, 0.0),
        ..Default::default()
    });
}