# Mapping runtime

`server/es_mapper_server.py` and `embodied_splat/` are the Mapping-owned mapper implementation. The engine is a
vendored runtime implementation, not a separate checkout. Its provenance and local path patches are in
`vendor_manifest.toml`; model files resolve through the internal `embodied_splat/pretrained ? Exp/assets/weights`
link. Generated mapper snapshots belong in `Exp/runs/<run>/artifacts`, never in this source tree.
