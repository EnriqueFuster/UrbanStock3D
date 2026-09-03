from datetime import date

import pytest
from pydantic import AnyHttpUrl, ValidationError

from urbanstock3d.domain.cadastre import (
    CadastralBuilding,
    CadastralBuildingPart,
    CadastralIdentity,
    ConstructionPeriod,
    FacadeDocument,
    Footprint2D,
    OfficialArea,
)


def sample_footprint() -> Footprint2D:
    return Footprint2D(
        crs="EPSG:25830",
        rings=(
            (
                (724385.52, 4372941.57),
                (724406.57, 4372941.57),
                (724406.57, 4372966.60),
                (724385.52, 4372941.57),
            ),
        ),
        estimated_accuracy_m=0.1,
        reference="footPrint",
    )


def test_building_model_represents_observed_catastro_attributes() -> None:
    footprint = sample_footprint()
    building = CadastralBuilding(
        identity=CadastralIdentity(
            cadastral_root_id="4531917yj2743b",
            local_id="4531917YJ2743B",
            namespace="ES.SDGC.BU",
        ),
        footprint=footprint,
        parts=(
            CadastralBuildingPart(
                local_id="4531917YJ2743B_part1",
                floors_above_ground=10,
                floors_below_ground=0,
                height_below_ground_m=0,
                footprint=footprint,
            ),
        ),
        condition="functional",
        current_use="1_residential",
        construction_period=ConstructionPeriod(
            beginning=date(1952, 1, 1),
            end=date(1952, 1, 1),
        ),
        building_units=17,
        dwellings=16,
        official_area=OfficialArea(value_m2=2698, reference="grossFloorArea"),
        facade_document=FacadeDocument(
            url=AnyHttpUrl(
                "http://ovc.catastro.meh.es/OVCServWeb/OVCWcfLibres/"
                "OVCFotoFachada.svc/RecuperarFotoFachadaGet"
                "?ReferenciaCatastral=4531917YJ2743B"
            ),
            media_format="jpeg",
            source_status="NotOfficial",
        ),
    )

    assert building.identity.cadastral_root_id == "4531917YJ2743B"
    assert building.identity.building_id == "ES.SDGC.BU.4531917YJ2743B"
    assert building.parts[0].floors_above_ground == 10
    assert building.facade_document is not None


def test_optional_cadastral_attributes_may_be_absent() -> None:
    building = CadastralBuilding(
        identity=CadastralIdentity(
            cadastral_root_id="4531917YJ2743B",
            local_id="4531917YJ2743B",
            namespace="ES.SDGC.BU",
        ),
        footprint=sample_footprint(),
    )

    assert building.current_use is None
    assert building.facade_document is None
    assert building.parts == ()


def test_cadastral_root_must_have_fourteen_characters() -> None:
    with pytest.raises(ValidationError):
        CadastralIdentity(
            cadastral_root_id="too-short",
            local_id="example",
            namespace="ES.SDGC.BU",
        )


def test_negative_registered_counts_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CadastralBuilding(
            identity=CadastralIdentity(
                cadastral_root_id="4531917YJ2743B",
                local_id="4531917YJ2743B",
                namespace="ES.SDGC.BU",
            ),
            footprint=sample_footprint(),
            dwellings=-1,
        )
