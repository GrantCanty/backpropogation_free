# Configurations

Reproducible experiment configurations are stored here as JSON files. The
8x8-digits benchmark uses scikit-learn's bundled dataset and never downloads
data; its checked-in default is in `digits_8x8.json`.
The matched no-backprop/BPTT run is in `digits_systems_comparison.json`.
Milestone 6 quality and forgetting-factor defaults are in `milestone6.json`;
the lazy 8x8/28x28 systems benchmark is in `scaling.json`.
The recurring-regime permanent-capacity and expanded RLS-factor capstone is in
`memory_capstone.json`.
The disjoint-development-seed, matched online-Adam comparison is in
`memory_backprop_comparison.json`.
The paired recurrent/pixel/fixed-convolution frontend study is in
`frontends.json`.
The forward-only masked latent-prediction study is in `predictive.json`.
The stable-backbone predictive-surprise recruitment study and its aligned,
lagged, and unused-predictor controls are in `predictive_surprise.json`.
The fixed contrast-polarity geometry ablation is in `polarity.json`.
The recurring multi-transformation benchmark is in `drift_suite.json`; it uses
signed-magnitude convolution as its primary spatial baseline.
The factor-free complementary-memory defaults are in
`factor_free_memory.json`; it includes fixed-key ungated, entropy-gated, and
RLS-leverage-gated controls, probationary frozen keys, and adaptive key-value
maturity variants. It also includes dense, top-k, winner-only, and normalized
local-responsibility controls plus evidence-managed 32/16/8-slot candidate
pools and fast-value/consolidated-fast-slow value ablations.
