# Curated production record

The original run used a staged CPU/GPU pipeline with retained checkpoints. The stages were:

`prepare -> symmetric features -> hand detector cross-fit -> pair aggregation -> family rankers -> evidence fusion -> validation -> inference -> schema checks`

The selected v53 run added a narrowly scoped depth-three soft-play MAP component to the v50 base; v51 kept v50 evidence and behavior while replacing 1,486 audited risk cells. The root project retains the complete historical scripts and large caches; this public package keeps the human-readable configuration and integrity record while the artifact directory remains the exact selected-file source.

The notebook's method cells are designed to be executable with the competition data, while the hash-verified replay is the authoritative route for recreating the two submitted files.

