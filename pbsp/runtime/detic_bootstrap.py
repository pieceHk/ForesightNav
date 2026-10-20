"""Register only the missing training-data imports required by Detic inference."""

import sys
import types


def install_inference_import_shims(detic_root):
    """Shim absent training-only imports in the pruned Detic vendor tree."""

    groups = (
        (
            detic_root / "third_party/CenterNet2/centernet/data",
            "centernet.data",
            {
                "datasets.objects365": {"categories_v1": ()},
                "datasets.coco": {"_PREDEFINED_SPLITS_COCO": {}},
                "datasets.nuimages": {},
            },
        ),
        (
            detic_root / "detic/data",
            "detic.data",
            {
                "datasets.lvis_v1": {},
                "datasets.imagenet": {},
                "datasets.cc": {},
                "datasets.objects365": {},
                "datasets.oid": {},
                "datasets.coco_zeroshot": {},
            },
        ),
    )
    installed = []
    for source_dir, package_root, child_specs in groups:
        if source_dir.is_dir():
            continue
        package_names = {package_root}
        for relative_name in child_specs:
            pieces = relative_name.split(".")
            for length in range(1, len(pieces)):
                package_names.add(package_root + "." + ".".join(pieces[:length]))
        for name in sorted(package_names, key=lambda item: item.count(".")):
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = []
                module.__scsp_inference_stub__ = True
                sys.modules[name] = module
                installed.append(name)
            if "." in name:
                parent_name, attribute = name.rsplit(".", 1)
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, attribute, sys.modules[name])
        for relative_name, attributes in child_specs.items():
            name = package_root + "." + relative_name
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__scsp_inference_stub__ = True
                sys.modules[name] = module
                installed.append(name)
            for attribute, value in attributes.items():
                setattr(sys.modules[name], attribute, value)
            parent_name, attribute = name.rsplit(".", 1)
            setattr(sys.modules[parent_name], attribute, sys.modules[name])
    try:
        __import__("timm.models.convnext")
    except ModuleNotFoundError as exc:
        if exc.name != "timm.models.convnext":
            raise
        name = (
            "detic.modeling.backbone.timm"
        )
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__scsp_inference_stub__ = True
            sys.modules[name] = module
            installed.append(name)
    return tuple(installed)
