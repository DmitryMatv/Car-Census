from car_census.mmr.trafficeye import parse_mmr_response


def test_parse_mmr_response_reads_nested_value_shapes() -> None:
    payload = {
        "combinations": [
            {
                "roadUsers": [
                    {
                        "mmr": {
                            "make": {"value": "Audi", "score": 0.91},
                            "model": {"value": "A4", "score": 0.82},
                        }
                    }
                ]
            }
        ]
    }
    result = parse_mmr_response(payload)
    assert result.make == "Audi"
    assert result.model == "A4"
    assert result.make_confidence == 0.91
    assert result.model_confidence == 0.82
