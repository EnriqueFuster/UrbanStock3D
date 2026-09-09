from scripts.prepare_roofer_input import projected_feature_collection


def test_projected_feature_collection_transforms_and_identifies_building() -> None:
    building = {
        "type": "Feature",
        "id": "building-1",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [-0.3912, 39.4769],
                    [-0.3911, 39.4769],
                    [-0.3911, 39.4770],
                    [-0.3912, 39.4769],
                ]
            ],
        },
    }

    collection = projected_feature_collection(building)

    feature = collection["features"][0]
    first_position = feature["geometry"]["coordinates"][0][0]
    assert collection["crs"]["properties"]["name"].endswith("25830")
    assert feature["properties"]["building_id"] == "building-1"
    assert 700_000 < first_position[0] < 750_000
    assert 4_350_000 < first_position[1] < 4_400_000


def test_projected_feature_collection_supports_multipolygon() -> None:
    building = {
        "id": "building-2",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [[[-0.39, 39.47], [-0.38, 39.47], [-0.39, 39.47]]],
                [[[-0.37, 39.48], [-0.36, 39.48], [-0.37, 39.48]]],
            ],
        },
    }

    collection = projected_feature_collection(building)

    assert collection["features"][0]["geometry"]["type"] == "MultiPolygon"
