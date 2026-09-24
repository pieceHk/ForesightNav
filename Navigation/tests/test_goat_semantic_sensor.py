from pathlib import Path
import sys

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
EVALUATOR_ROOT = ROOT / "Navigation/runtime/evaluators/goat"
sys.path.insert(0, str(EVALUATOR_ROOT))

from config_utils import disable_ground_truth_semantic_sensor


def _config():
    return OmegaConf.create(
        {
            "habitat": {
                "simulator": {
                    "agents": {
                        "main_agent": {
                            "sim_sensors": {
                                "rgb_sensor": {"height": 480},
                                "depth_sensor": {"height": 480},
                                "semantic_sensor": {"height": 480},
                            }
                        }
                    }
                }
            }
        }
    )


def test_disabled_semantics_remove_only_the_gt_render_sensor():
    config = _config()

    assert disable_ground_truth_semantic_sensor(config, "disabled")
    sensors = config.habitat.simulator.agents.main_agent.sim_sensors
    assert "semantic_sensor" not in sensors
    assert {"rgb_sensor", "depth_sensor"} <= set(sensors)
    assert not disable_ground_truth_semantic_sensor(config, "disabled")


def test_ground_truth_mode_keeps_the_semantic_sensor():
    config = _config()

    assert not disable_ground_truth_semantic_sensor(config, "ground_truth")
    assert "semantic_sensor" in config.habitat.simulator.agents.main_agent.sim_sensors
