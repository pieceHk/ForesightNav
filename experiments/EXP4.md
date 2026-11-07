# Eq.(13) BLP mechanism ablation

`suffix.mode: blp_eq13` replaces only the current-task candidate order. The four
`blp_variant` values are `nearest`, `suffix_only`, `myopic`, and `full`. Projection,
spatial blacklist reconciliation, verification, category cooldown and navigation
use the native pipeline. Previous modes retain their original planning path.

`pbsp/navigation/blp.py` builds version-bound raw-rank top-five candidate layers
and solves the success/failure recursion in metres. It caches geodesics and
memoizes `(task, location, removed_mask)`. The last task uses the same recursion.
The candidate evidence bank counts visible camera bearings in twelve bins.
Frontier pixels are converted through the native episode GPS transform, snapped
to the navmesh, and queried with the native strict geodesic function.

`semantic_score` is the selected Gaussian's codebook-weighted text similarity.
It is not an average over neighbouring support Gaussians; support counts are
used separately by the unchanged query engine. Exp4 uses the existing field.

Prepare native identities, fixed folds and configuration files:

```bash
python -m experiments.exp4_prepare
python -m unittest experiments.test_blp -v
```

Preparation writes `results/exp4/meta/` and a machine-local runtime configuration.
Use the two original PBSP Python environments in `runtime.local.yaml`.
The shadow calibration configuration keeps `ordered_suffix_mvp` in control:

```bash
python run.py --config configs/experiments/exp4_calibration.yaml \
  --runtime runtime.exp4.local.yaml --gpus 2,2 --scene SCENE_ID
```

Calibration snapshots contain the complete candidate/distance input for offline
replay. `blp_events.jsonl` distinguishes first appearances, checked candidates,
final verification outcomes, and empty-activation exploration lengths. A failed
verification frame is not an independent negative label: checks are finalized
on confirmation or abandonment. GT diagnostics reuse copies of the pinned GOAT
distance/success measures at a virtual candidate position without moving the
real agent. GT values never enter the planner or the self-label fit.

Shadow-only provisional parameters are `a=10`, `b0=-2.5`, `pi=0.3`, `c_exp=10 m`.
The design supplies the first three; the last is an explicit plumbing placeholder
until empty-activation path measurements are available. None are evaluation
parameters. Active evaluation refuses an absent or unfrozen calibration file.

`experiments/blp_calibration.py` provides positive-slope maximum-likelihood fits,
episode bootstrap, confusion/reliability diagnostics and exact decision replay.
One-class or unbounded fits raise an error rather than emitting invented
parameters. Candidate priors count distinct checked spatial identities.

Recovery waypoints are tagged as `navmesh_override` at the actual verification
call. Their events are retained as `noncandidate_verification` and excluded from
the candidate belief fit because they have no query semantic score. A protected
query candidate missing from a newer batch uses its recorded query evidence,
with explicit historical provenance; an unknown query candidate raises a logged
calibration error for review. No semantic score is fabricated for a recovery point.

The one-metre switching hysteresis and the existing category cooldown apply to
all four variants. Thus Nearest's unconstrained proposal has zero deviation from
nearest, while its realized active candidate can be held by either mechanism.
Reports must distinguish proposal, hysteresis and final dispatch, including any
nonzero realized Nearest deviation. This conflict in the supplied design requires
an explicit reporting decision before the evaluation comparison.

`experiments.exp4_regression` replays passive planner captures from three fixed
native episodes. It compares all planning values and orders, excluding wall-clock
timings. Original capture inputs and source snapshots are retained with results.
This is a matched-input stepwise regression, not a claim of bitwise-identical
independent GPU simulation rollouts.
