# LagMemo runtime sources

This directory holds the default navigation backend. PBSP policy and component interfaces are in the root `pbsp/` package.

- `src/lagmemo/` contains the sensor, perception and local-map adapters used by navigation.
- `src/third_party/` retains MobileCLIP, SEEM and frontier source. Detectron2 is installed from its fixed complete source archive; Habitat and SuperGlue are prepared as checked external dependencies.
- `project/` contains evaluator configuration and Habitat glue.
- `data/` holds the goal categories and compatibility data links.
- `datadump/` receives compatibility-entry output.

The native factory supplies checkpoint and tokenizer paths through resource objects. Compatibility entries also support the internal `Exp/assets/weights` layout. The [vendor manifest](../vendor_manifest.toml) records source ownership and retained dependencies; the [installation guide](../../../docs/installation.md) gives the preparation commands and license links.
