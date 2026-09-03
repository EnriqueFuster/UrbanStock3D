import pytest
from pydantic import ValidationError

from urbanstock3d.domain.cadastre import CoordinateQuery, RefcatQuery


@pytest.mark.parametrize(
    ("value", "expected_root"),
    [
        ("4531917YJ2743B", "4531917YJ2743B"),
        ("4531917YJ2743B0001", "4531917YJ2743B"),
        ("4531917YJ2743B0001AB", "4531917YJ2743B"),
        (" 4531917yj2743b ", "4531917YJ2743B"),
    ],
)
def test_refcat_query_normalizes_documented_lengths(
    value: str,
    expected_root: str,
) -> None:
    query = RefcatQuery(value=value)

    assert query.cadastral_root_id == expected_root


@pytest.mark.parametrize(
    "value",
    [
        "4531917YJ2743",
        "4531917YJ2743B000",
        "4531917YJ2743B0001A",
        "4531917YJ2743!",
    ],
)
def test_refcat_query_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValidationError):
        RefcatQuery(value=value)


@pytest.mark.parametrize(
    ("longitude", "latitude"),
    [(181, 0), (-181, 0), (0, 91), (0, -91)],
)
def test_coordinate_query_rejects_values_outside_wgs84_bounds(
    longitude: float,
    latitude: float,
) -> None:
    with pytest.raises(ValidationError):
        CoordinateQuery(longitude=longitude, latitude=latitude)
