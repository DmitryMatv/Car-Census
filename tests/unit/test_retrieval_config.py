import pytest
from pydantic import ValidationError


def test_retrieval_defaults_are_conservative(default_config) -> None:
    assert default_config.project.retrieval_cache_dir.name == ".mmr-cache"
    assert default_config.mmr.retrieval_mode == "shadow"
    assert default_config.mmr.retrieval_embedding_distance_threshold == 0.02
    assert default_config.mmr.retrieval_phash_max_hamming_distance == 4
    assert default_config.mmr.retrieval_min_neighbors == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retrieval_mode", "local_classifier"),
        ("retrieval_embedding_distance_threshold", -0.01),
        ("retrieval_embedding_distance_threshold", 2.01),
        ("retrieval_phash_max_hamming_distance", -1),
        ("retrieval_phash_max_hamming_distance", 65),
        ("retrieval_min_neighbors", 0),
    ],
)
def test_retrieval_config_rejects_unsafe_values(
    config_factory, field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        config_factory({"mmr": {field: value}})
