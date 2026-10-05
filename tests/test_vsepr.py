"""Tests for VSEPR molecular geometry renderer."""

import pytest
from mcq.rendering.vsepr_renderer import is_vsepr_molecule, generate_vsepr_svg, _VSEPR_REGISTRY


@pytest.mark.parametrize(
    "molecule,expected_sn,expected_geom",
    [
        ("BeCl2", 2, "linear"),
        ("BF3", 3, "trigonal_planar"),
        ("SO2", 3, "bent_sn3"),
        ("CH4", 4, "tetrahedral"),
        ("NH3", 4, "trigonal_pyramidal"),
        ("H2O", 4, "bent_sn4"),
        ("PCl5", 5, "trigonal_bipyramidal"),
        ("SF4", 5, "seesaw"),
        ("ClF3", 5, "t_shaped"),
        ("XeF2", 5, "linear_sn5"),
        ("SF6", 6, "octahedral"),
        ("BrF5", 6, "square_pyramidal"),
        ("XeF4", 6, "square_planar"),
    ],
)
def test_vsepr_detection_and_svg_generation(molecule, expected_sn, expected_geom):
    assert is_vsepr_molecule(molecule)
    info = _VSEPR_REGISTRY[molecule.lower()]
    assert info["sn"] == expected_sn
    assert info["geometry"] == expected_geom

    svg = generate_vsepr_svg(info)
    assert svg.startswith("<svg")
    assert svg.strip().endswith("</svg>")
    assert info["central"] in svg


def test_vsepr_stereochemical_notation():
    # Tetrahedral CH4 must include solid wedges and dashed wedges
    ch4_svg = generate_vsepr_svg(_VSEPR_REGISTRY["ch4"])
    assert "<polygon" in ch4_svg, "Tetrahedral must have solid wedge"
    assert "<line" in ch4_svg, "Tetrahedral must have dashed wedge"

    # Ammonia NH3 must include lone pair orbital lobe
    nh3_svg = generate_vsepr_svg(_VSEPR_REGISTRY["nh3"])
    assert "<ellipse" in nh3_svg, "NH3 must have lone pair ellipse"
    assert "<circle" in nh3_svg, "NH3 must have lone pair electrons"

    # PCl5 must have axial bonds and equatorial wedges
    pcl5_svg = generate_vsepr_svg(_VSEPR_REGISTRY["pcl5"])
    assert "<polygon" in pcl5_svg
    assert "<line" in pcl5_svg
