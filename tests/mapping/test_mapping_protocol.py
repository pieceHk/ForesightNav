import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Mapping" / "runtime" / "server"))

from es_mapper_server import ESMapperServer


class FakeLiveMapper:
    def query_with_records(self, category, top_k):
        candidates = [[1.0, 0.0, 2.0]]
        return candidates, [
            {
                "position": candidates[0],
                "raw_rank": 0,
                "semantic_score": 0.5,
                "support_count": 4,
                "support_score": 1.0,
                "map_revision": 20,
            }
        ]

    def query_batch_with_records(self, categories, top_k):
        candidates = {category: [[1.0, 0.0, 2.0]] for category in categories}
        records = {
            category: [
                {
                    "position": candidates[category][0],
                    "raw_rank": 0,
                    "semantic_score": 0.5,
                    "support_count": 4,
                    "support_score": 1.0,
                    "map_revision": 20,
                }
            ]
            for category in categories
        }
        return candidates, records

    def get_status(self):
        return {"snap": "live", "nav_steps": 20}


def _server():
    server = ESMapperServer.__new__(ESMapperServer)
    server.top_k = 10
    server._live_mode = True
    server._live_mapper = FakeLiveMapper()
    server._mapper = None
    return server


def test_single_query_keeps_legacy_candidates_and_adds_records():
    response = _server().handle_query(0, "chair", 10)
    assert response["candidates"] == [[1.0, 0.0, 2.0]]
    assert response["num_candidates"] == 1
    assert response["candidate_records"][0]["map_revision"] == 20


def test_batch_query_keeps_legacy_results_and_adds_records():
    response = _server().handle_batch_query(["chair", "bed"], 10)
    assert response["results"]["chair"] == [[1.0, 0.0, 2.0]]
    assert response["candidate_records_by_category"]["bed"][0]["raw_rank"] == 0
