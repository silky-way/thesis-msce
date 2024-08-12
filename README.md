# Modelling conflicts among cyclists on cycle
highways
## Thesis submitted for the degree of Master of Science in Mobility and Supply Chain Engineering at the KU Leuven


Due to the growth in cyclists on bicycle paths with more complex interactions because
of increased heterogeneity, additional research is necessary to make substantiated
decisions around bicycle infrastructure and legislation. Despite great attention
from influential bodies like the European Commission, research often lacks coherent
analysis towards preventing conflicts on separated cycle lanes. This thesis explores
the factors driving tension and discomfort on bicycle highways and other comparable
infrastructure where there is little to no interaction with motorized traffic. Not the
conflicts themselves, but the macroscopic trends that could lead to conflict-prone
situations or relevant discomfort are researched on a link level, where most state-of-
the-art models focus on conflict-prone locations on a node-level like crossroads. The
microscopic Social Force Model adapted for bicycle use was implemented in Rust’s
programming language utilising the game engine Bevy from the ground up. This
model determines changes in position based on forces from the environment and
other users, combining all forces and using Newton’s second law of motion to find the
acceleration of each entity. The position change can then be found using the numerical
Verlet integration method frequently used to calculate trajectories of particles. The
basic Social Force Model equations were used to set up this technique, enriched
with additional functionalities to create a smooth flow of users. With the results of
this simulation, a comparative analysis for six scenarios differing only in user mix
distributions showed counterintuitive effects where the most heterogeneous scenario
scored high on four out of five metrics that were analysed. A visual representation
aided in linking these insights to other known traffic engineering concepts and together
with the thoroughly documented code offers a foundation for future studies
