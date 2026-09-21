# Curated production record

The original run used a staged CPU/GPU pipeline with retained checkpoints. The stages were:

`prepare -> symmetric features -> hand detector cross-fit -> pair aggregation -> family rankers -> evidence fusion -> validation -> inference -> schema checks`

The selected v53 run added a narrowly scoped depth-three soft-play MAP component to the v50 base; v51 kept v50 evidence and behavior while replacing 1,486 audited risk cells. The root project retains the complete historical scripts and large caches; this public package keeps the human-readable configuration and integrity record while the artifact directory remains the exact selected-file source.

The notebook executes a complete compact feature/training/inference path on the competition data and saves its fresh candidate. The hash-verified selected-file assembly is the authoritative route for recreating the two submitted files: v51 and v53 were historical checkpoint variants of v50, so a new fit from raw tables alone is not expected to reproduce their bytes. The retained production source, configurations, and checkpoint manifests document that boundary.
