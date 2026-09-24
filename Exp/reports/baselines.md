# Regression Baselines

## HM3Dv1 ObjectNav v1

- Artifact: `20260809_HM3Dv1_v1_consolidated_0000_0278`
- Episodes: 279
- SR: 37.28%
- SPL: 15.12%
- SoftSPL: 22.88%
- Mean steps: 276.69

## Acceptance gates

1. Unit policies must match the reference implementation on identical inputs.
2. Focused matched episodes must not lose successful outcomes.
3. A full result cannot be compared until dataset, split, semantic source,
   maximum steps, seeds, and episode indices are identical.
4. Episode-reset RSS floor must remain bounded; pinned-host cache is included.
5. GPU memory must return to the established online-mapping plateau after reset.
