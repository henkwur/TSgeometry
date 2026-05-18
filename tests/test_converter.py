from pathlib import Path

import laspy
from PIL import Image

from image2las.converter import ConversionConfig, convert_image_to_las


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
    assert source_path.exists()

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
    assert source_path.exists()

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
