from pathlib import Path

import laspy
import numpy as np
import pytest
import shapefile
from PIL import Image

from image2las.converter import (
    ConversionConfig,
    _extract_offset_wgs84,
    _extract_plot_angle,
    _extract_plot_position,
    _read_envi_metadata,
    _rotate_xy_from_north_clockwise,
    _write_offset_shapefile,
    _wgs84_to_rd_new,
    convert_image_to_las,
)


def test_convert_image_to_las(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.las"

    image = Image.new("L", (2, 2))
    image.putdata([0, 64, 128, 255])
    image.save(input_path)

    result = convert_image_to_las(
        ConversionConfig(
            input_path=input_path,
            output_path=output_path,
            x_scale=1.0,
            y_scale=1.0,
            z_scale=1.0,
        )
    )

    assert result == output_path
    assert output_path.exists()

    las = laspy.read(output_path)
    assert len(las.x) == 4
    assert list(las.z) == [0.0, 64.0, 128.0, 255.0]


def test_convert_envi_hyperspectral_to_las(tmp_path: Path) -> None:
    source_path = Path("testimage/Fused-VNIR-Zuid-20260116-143627-001.cmb.hdr")
    raw_path = Path("testimage/Fused-VNIR-Zuid-20260116-143627-001.cmb.raw")
    assert source_path.exists()
    if not raw_path.exists():
        pytest.skip("ENVI raw test file is not available in this workspace")

    output_path = tmp_path / "fused-converted.las"

    result = convert_image_to_las(
        ConversionConfig(
            input_path=source_path,
            output_path=output_path,
            band_index=93,
        )
    )

    assert result == output_path
    assert output_path.exists()

    las = laspy.read(output_path)
    assert len(las.x) > 0
    assert len(las.x) == len(las.y) == len(las.z)


def test_convert_envi_with_coordinates(tmp_path: Path) -> None:
    source_path = Path("testimage/Fused-VNIR-Zuid-20260116-143627-001.cmb.hdr")
    raw_path = Path("testimage/Fused-VNIR-Zuid-20260116-143627-001.cmb.raw")
    assert source_path.exists()
    if not raw_path.exists():
        pytest.skip("ENVI raw test file is not available in this workspace")

    output_path = tmp_path / "fused-with-coords.las"

    result = convert_image_to_las(
        ConversionConfig(
            input_path=source_path,
            output_path=output_path,
            use_envi_coordinates=True,
        )
    )

    assert result == output_path
    assert output_path.exists()

    las = laspy.read(output_path)
    assert len(las.x) > 0
    assert len(las.x) == len(las.y) == len(las.z)
    
    # Verify the coordinates are extracted from ENVI bands (0-9999 meter range typical)
    assert las.x.min() >= -10000 and las.x.max() <= 10000
    assert las.y.min() >= -10000 and las.y.max() <= 10000
    assert las.z.min() >= -10000 and las.z.max() <= 10000


def test_extract_wgs84_offset_from_envi_extrainfo_and_convert_to_rd() -> None:
    source_path = Path("testimage/Fused-VNIR-Zuid-20260116-143627-001.cmb.hdr")
    assert source_path.exists()

    metadata = _read_envi_metadata(source_path)
    offset_wgs84 = _extract_offset_wgs84(metadata)
    assert offset_wgs84 is not None

    rd_x, rd_y = _wgs84_to_rd_new(offset_wgs84[0], offset_wgs84[1])

    # Typical RD New range in the Netherlands.
    assert 0.0 < rd_x < 300000.0
    assert 300000.0 < rd_y < 650000.0


def test_write_offset_shapefile(tmp_path: Path) -> None:
    source_name = "Fused-VNIR-Zuid-20260116-143627-001.cmb"
    shp_path = tmp_path / f"{source_name}_Offset.shp"

    _write_offset_shapefile(shp_path, 160000.123, 460000.456, source_name)

    assert shp_path.exists()
    assert shp_path.with_suffix(".shx").exists()
    assert shp_path.with_suffix(".dbf").exists()
    assert shp_path.with_suffix(".prj").exists()

    reader = shapefile.Reader(str(shp_path))
    shape = reader.shape(0)
    record = reader.record(0)

    assert shape.shapeType == shapefile.POINT
    assert shape.points[0][0] == pytest.approx(160000.123, abs=1e-3)
    assert shape.points[0][1] == pytest.approx(460000.456, abs=1e-3)
    assert record[0] == source_name


def test_extract_plot_angle_from_extrainfo() -> None:
    metadata = {
        "extrainfo": "{ OffsetLat:51.0, OffsetLong:5.0, PlotAngle:333.91602892904 }"
    }
    angle = _extract_plot_angle(metadata)
    assert angle == pytest.approx(333.91602892904)


def test_rotate_xy_from_north_clockwise() -> None:
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])

    x0, y0 = _rotate_xy_from_north_clockwise(x, y, 0.0)
    assert np.allclose(x0, x)
    assert np.allclose(y0, y)

    x90, y90 = _rotate_xy_from_north_clockwise(np.array([0.0]), np.array([1.0]), 90.0)
    assert x90[0] == pytest.approx(1.0, abs=1e-9)
    assert y90[0] == pytest.approx(0.0, abs=1e-9)


def test_extract_plot_position_from_extrainfo() -> None:
    metadata = {
        "extrainfo": "{ PlotPositionX:1.41544087091461, PlotPositionY:-2.64541777968407 }"
    }
    position = _extract_plot_position(metadata)
    assert position is not None
    assert position[0] == pytest.approx(1.41544087091461)
    assert position[1] == pytest.approx(-2.64541777968407)


def test_plot_position_shifts_output_from_rd_origin(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.las"

    image = Image.new("L", (1, 1))
    image.putdata([0])
    image.save(input_path)

    from image2las import converter

    original_read_metadata = converter._read_envi_metadata
    original_extract_offset = converter._extract_offset_wgs84
    original_wgs84_to_rd = converter._wgs84_to_rd_new
    try:
        converter._read_envi_metadata = lambda _p: {"extrainfo": "{ PlotPositionX:3.0, PlotPositionY:-4.0 }"}
        converter._extract_offset_wgs84 = lambda _m: (51.0, 5.0)
        converter._wgs84_to_rd_new = lambda _lat, _lon: (1000.0, 2000.0)

        convert_image_to_las(
            ConversionConfig(
                input_path=input_path,
                output_path=output_path,
                invert_y=False,
                use_envi_coordinates=False,
            )
        )
    finally:
        converter._read_envi_metadata = original_read_metadata
        converter._extract_offset_wgs84 = original_extract_offset
        converter._wgs84_to_rd_new = original_wgs84_to_rd

    las = laspy.read(output_path)
    assert las.x[0] == pytest.approx(1003.0, abs=1e-6)
    assert las.y[0] == pytest.approx(1996.0, abs=1e-6)

    original_offset = tmp_path / "input_Offset.shp"
    translated_offset = tmp_path / "input_plotoffset.shp"
    offsets_txt = tmp_path / "input_offsets.txt"
    assert original_offset.exists()
    assert translated_offset.exists()
    assert offsets_txt.exists()

    original_reader = shapefile.Reader(str(original_offset))
    translated_reader = shapefile.Reader(str(translated_offset))
    original_point = original_reader.shape(0).points[0]
    translated_point = translated_reader.shape(0).points[0]

    assert original_point[0] == pytest.approx(1000.0, abs=1e-6)
    assert original_point[1] == pytest.approx(2000.0, abs=1e-6)
    assert translated_point[0] == pytest.approx(1003.0, abs=1e-6)
    assert translated_point[1] == pytest.approx(1996.0, abs=1e-6)

    lines = offsets_txt.read_text(encoding="utf-8").strip().splitlines()
    assert "Georeference calculation report" in lines
    assert "- OffsetLat/OffsetLong source: ENVI 'extrainfo' (OffsetLat/OffsetLong)" in lines
    assert "- PlotPositionX/PlotPositionY source: ENVI 'extrainfo' (PlotPositionX/PlotPositionY)" in lines
    assert "   PlotOffset.X = 1000.000000 + 3.000000000000 = 1003.000000" in lines
    assert "   PlotOffset.Y = 2000.000000 + -4.000000000000 = 1996.000000" in lines
    assert "Name\tX\tY" in lines
    assert "Offset\t1000.000000\t2000.000000" in lines
    assert "PlotOffset\t1003.000000\t1996.000000" in lines


def test_plot_enter_exit_are_exported_to_rdnew_and_reported(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.las"

    image = Image.new("L", (1, 1))
    image.putdata([0])
    image.save(input_path)

    from image2las import converter

    original_read_metadata = converter._read_envi_metadata
    original_extract_offset = converter._extract_offset_wgs84
    original_wgs84_to_rd = converter._wgs84_to_rd_new
    try:
        converter._read_envi_metadata = lambda _p: {
            "extrainfo": "{ PlotEnterLat:51.100000, PlotEnterLong:5.200000, PlotExitLat:51.300000, PlotExitLong:5.400000 }"
        }
        converter._extract_offset_wgs84 = lambda _m: None
        converter._wgs84_to_rd_new = lambda lat, lon: (lon * 1000.0, lat * 1000.0)

        convert_image_to_las(
            ConversionConfig(
                input_path=input_path,
                output_path=output_path,
                invert_y=False,
                use_envi_coordinates=False,
            )
        )
    finally:
        converter._read_envi_metadata = original_read_metadata
        converter._extract_offset_wgs84 = original_extract_offset
        converter._wgs84_to_rd_new = original_wgs84_to_rd

    plot_enter = tmp_path / "input_PlotEnter.shp"
    plot_exit = tmp_path / "input_PlotExit.shp"
    offsets_txt = tmp_path / "input_offsets.txt"
    assert plot_enter.exists()
    assert plot_exit.exists()
    assert offsets_txt.exists()

    enter_reader = shapefile.Reader(str(plot_enter))
    exit_reader = shapefile.Reader(str(plot_exit))
    enter_point = enter_reader.shape(0).points[0]
    exit_point = exit_reader.shape(0).points[0]

    assert enter_point[0] == pytest.approx(5200.0, abs=1e-6)
    assert enter_point[1] == pytest.approx(51100.0, abs=1e-6)
    assert exit_point[0] == pytest.approx(5400.0, abs=1e-6)
    assert exit_point[1] == pytest.approx(51300.0, abs=1e-6)

    lines = offsets_txt.read_text(encoding="utf-8").strip().splitlines()
    assert "- PlotEnterLat/PlotEnterLong source: ENVI 'extrainfo' (PlotEnterLat/PlotEnterLong)" in lines
    assert "- PlotExitLat/PlotExitLong source: ENVI 'extrainfo' (PlotExitLat/PlotExitLong)" in lines
    assert "PlotEnter\t5200.000000\t51100.000000" in lines
    assert "PlotExit\t5400.000000\t51300.000000" in lines


def test_skip_las_write_still_generates_reports(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.las"

    image = Image.new("L", (1, 1))
    image.putdata([0])
    image.save(input_path)

    from image2las import converter

    original_read_metadata = converter._read_envi_metadata
    original_extract_offset = converter._extract_offset_wgs84
    original_wgs84_to_rd = converter._wgs84_to_rd_new
    try:
        converter._read_envi_metadata = lambda _p: {
            "extrainfo": "{ PlotPositionX:3.0, PlotPositionY:-4.0 }"
        }
        converter._extract_offset_wgs84 = lambda _m: (51.0, 5.0)
        converter._wgs84_to_rd_new = lambda _lat, _lon: (1000.0, 2000.0)

        convert_image_to_las(
            ConversionConfig(
                input_path=input_path,
                output_path=output_path,
                invert_y=False,
                use_envi_coordinates=False,
                write_las=False,
            )
        )
    finally:
        converter._read_envi_metadata = original_read_metadata
        converter._extract_offset_wgs84 = original_extract_offset
        converter._wgs84_to_rd_new = original_wgs84_to_rd

    assert not output_path.exists()
    assert (tmp_path / "input_offsets.txt").exists()
    assert (tmp_path / "input_Offset.shp").exists()
    assert (tmp_path / "input_plotoffset.shp").exists()


def test_rotation_happens_after_plot_position_translation(tmp_path: Path) -> None:
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.las"

    image = Image.new("L", (2, 1))
    image.putdata([0, 0])
    image.save(input_path)

    from image2las import converter

    original_read_metadata = converter._read_envi_metadata
    original_extract_offset = converter._extract_offset_wgs84
    original_wgs84_to_rd = converter._wgs84_to_rd_new
    try:
        converter._read_envi_metadata = lambda _p: {
            "extrainfo": "{ PlotPositionX:3.0, PlotPositionY:-4.0, PlotAngle:90.0 }"
        }
        converter._extract_offset_wgs84 = lambda _m: (51.0, 5.0)
        converter._wgs84_to_rd_new = lambda _lat, _lon: (1000.0, 2000.0)

        convert_image_to_las(
            ConversionConfig(
                input_path=input_path,
                output_path=output_path,
                invert_y=False,
                use_envi_coordinates=False,
            )
        )
    finally:
        converter._read_envi_metadata = original_read_metadata
        converter._extract_offset_wgs84 = original_extract_offset
        converter._wgs84_to_rd_new = original_wgs84_to_rd

    las = laspy.read(output_path)
    assert len(las.x) == 2

    # First point is at local (0, 0) and remains on the translated plot origin.
    assert las.x[0] == pytest.approx(1003.0, abs=1e-6)
    assert las.y[0] == pytest.approx(1996.0, abs=1e-6)
    # Second point is local (1, 0), rotated 90 degrees clockwise-from-north around the plot origin.
    assert las.x[1] == pytest.approx(1003.0, abs=1e-6)
    assert las.y[1] == pytest.approx(1995.0, abs=1e-6)


