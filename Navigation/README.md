# Navigation

`Navigation/src/tdonav_navigation` owns candidate execution, STOP boundary, recovery and action contracts.
`Navigation/runtime/lagmemo` is the owned Habitat/LagMemo evaluator runtime; `runtime/evaluators` holds GOAT and
HM3D entrypoints. It is selected only by a named internal entrypoint, never an external checkout path.
