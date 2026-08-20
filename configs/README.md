# Configurations

Reproducible experiment configurations are stored here as JSON files. The
8x8-digits benchmark uses scikit-learn's bundled dataset and never downloads
data; its checked-in default is in `digits_8x8.json`.
The matched no-backprop/BPTT run is in `digits_systems_comparison.json`.
Milestone 6 quality and forgetting-factor defaults are in `milestone6.json`;
the lazy 8x8/28x28 systems benchmark is in `scaling.json`.
The factor-free complementary-memory defaults are in
`factor_free_memory.json`; it includes fixed-key ungated, entropy-gated, and
RLS-leverage-gated controls plus adaptive key-value maturity variants.
