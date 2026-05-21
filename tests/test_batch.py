from pathlib import Path

from image2las import batch
from image2las.converter import ConversionConfig


def test_discover_envi_fused_hdr_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "001" / "001_20260521" / "data" / "ENVI-fused").mkdir(parents=True)
    (root / "001" / "001_20260521" / "data" / "ENVI-fused" / "plot001_VNIR.hdr").write_text("ENVI\n", encoding="utf-8")
    (root / "001" / "001_20260521" / "data" / "ENVI-fused" / "plot001_SWIR.hdr").write_text("ENVI\n", encoding="utf-8")
    (root / "001" / "001_20260521" / "data" / "other.hdr").write_text("ENVI\n", encoding="utf-8")

    found = batch.discover_envi_fused_hdr_files(root)

    assert len(found) == 2
    assert all(path.parent.name == "ENVI-fused" for path in found)


def test_convert_root_folder_writes_outputs_per_plot(tmp_path: Path) -> None:
    root = tmp_path / "root"
    in_a = root / "101" / "101_20260521" / "meta" / "ENVI-fused" / "plot101_VNIR.hdr"
    in_b = root / "202" / "202_20260521" / "meta" / "ENVI-fused" / "plot202_SWIR.hdr"
    in_a.parent.mkdir(parents=True)
    in_b.parent.mkdir(parents=True)
    in_a.write_text("ENVI\n", encoding="utf-8")
    in_b.write_text("ENVI\n", encoding="utf-8")

    output_root = tmp_path / "out"

    calls: list[tuple[Path, Path]] = []

    original_convert = batch.convert_image_to_las
    try:
        def fake_convert(config: ConversionConfig) -> Path:
            calls.append((config.input_path, config.output_path))
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            config.output_path.write_text("fake-las", encoding="utf-8")
            return config.output_path

        batch.convert_image_to_las = fake_convert

        def build_config(input_hdr: Path, output_las: Path) -> ConversionConfig:
            return ConversionConfig(input_path=input_hdr, output_path=output_las)

        result = batch.convert_root_folder(root, output_root, build_config)
    finally:
        batch.convert_image_to_las = original_convert

    assert len(result.failed) == 0
    assert len(result.converted) == 2
    assert (output_root / "101" / "plot101_VNIR.las").exists()
    assert (output_root / "202" / "plot202_SWIR.las").exists()
    assert len(calls) == 2


def test_convert_root_folder_orders_plots_numerically(tmp_path: Path) -> None:
    root = tmp_path / "root"
    in_1 = root / "1" / "1_20260521" / "meta" / "ENVI-fused" / "a_VNIR.hdr"
    in_2 = root / "2" / "2_20260521" / "meta" / "ENVI-fused" / "b_VNIR.hdr"
    in_10 = root / "10" / "10_20260521" / "meta" / "ENVI-fused" / "c_VNIR.hdr"
    in_1.parent.mkdir(parents=True)
    in_2.parent.mkdir(parents=True)
    in_10.parent.mkdir(parents=True)
    in_1.write_text("ENVI\n", encoding="utf-8")
    in_2.write_text("ENVI\n", encoding="utf-8")
    in_10.write_text("ENVI\n", encoding="utf-8")

    output_root = tmp_path / "out"
    seen_plot_ids: list[str] = []

    original_convert = batch.convert_image_to_las
    try:
        def fake_convert(config: ConversionConfig) -> Path:
            seen_plot_ids.append(config.output_path.parent.name)
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            config.output_path.write_text("ok", encoding="utf-8")
            return config.output_path

        batch.convert_image_to_las = fake_convert

        def build_config(input_hdr: Path, output_las: Path) -> ConversionConfig:
            return ConversionConfig(input_path=input_hdr, output_path=output_las)

        result = batch.convert_root_folder(root, output_root, build_config)
    finally:
        batch.convert_image_to_las = original_convert

    assert len(result.failed) == 0
    assert seen_plot_ids == ["1", "2", "10"]


def test_convert_root_folder_honors_cancel(tmp_path: Path) -> None:
    root = tmp_path / "root"
    in_a = root / "1" / "1_20260521" / "meta" / "ENVI-fused" / "a_VNIR.hdr"
    in_b = root / "2" / "2_20260521" / "meta" / "ENVI-fused" / "b_VNIR.hdr"
    in_a.parent.mkdir(parents=True)
    in_b.parent.mkdir(parents=True)
    in_a.write_text("ENVI\n", encoding="utf-8")
    in_b.write_text("ENVI\n", encoding="utf-8")

    output_root = tmp_path / "out"
    cancel_state = {"stop": False}

    original_convert = batch.convert_image_to_las
    try:
        def fake_convert(config: ConversionConfig) -> Path:
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            config.output_path.write_text("ok", encoding="utf-8")
            cancel_state["stop"] = True
            return config.output_path

        batch.convert_image_to_las = fake_convert

        def build_config(input_hdr: Path, output_las: Path) -> ConversionConfig:
            return ConversionConfig(input_path=input_hdr, output_path=output_las)

        result = batch.convert_root_folder(
            root,
            output_root,
            build_config,
            should_cancel=lambda: cancel_state["stop"],
        )
    finally:
        batch.convert_image_to_las = original_convert

    assert result.cancelled is True
    assert len(result.converted) == 1
    assert len(result.failed) == 0


def test_discover_envi_filtered_by_sensor_type(tmp_path: Path) -> None:
    root = tmp_path / "root"
    fused = root / "001" / "001_20260521" / "data" / "ENVI-fused"
    fused.mkdir(parents=True)
    (fused / "plot001_VNIR.hdr").write_text("ENVI\n", encoding="utf-8")
    (fused / "plot001_SWIR.hdr").write_text("ENVI\n", encoding="utf-8")

    vnir_only = batch.discover_envi_fused_hdr_files_filtered(root, include_vnir=True, include_swir=False)
    swir_only = batch.discover_envi_fused_hdr_files_filtered(root, include_vnir=False, include_swir=True)

    assert len(vnir_only) == 1
    assert "vnir" in vnir_only[0].stem.lower()
    assert len(swir_only) == 1
    assert "swir" in swir_only[0].stem.lower()


def test_convert_root_folder_respects_sensor_filters(tmp_path: Path) -> None:
    root = tmp_path / "root"
    fused = root / "1" / "1_20260521" / "meta" / "ENVI-fused"
    fused.mkdir(parents=True)
    (fused / "plot1_VNIR.hdr").write_text("ENVI\n", encoding="utf-8")
    (fused / "plot1_SWIR.hdr").write_text("ENVI\n", encoding="utf-8")

    output_root = tmp_path / "out"
    seen_inputs: list[str] = []

    original_convert = batch.convert_image_to_las
    try:
        def fake_convert(config: ConversionConfig) -> Path:
            seen_inputs.append(config.input_path.name)
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            config.output_path.write_text("ok", encoding="utf-8")
            return config.output_path

        batch.convert_image_to_las = fake_convert

        def build_config(input_hdr: Path, output_las: Path) -> ConversionConfig:
            return ConversionConfig(input_path=input_hdr, output_path=output_las)

        result = batch.convert_root_folder(
            root,
            output_root,
            build_config,
            include_vnir=False,
            include_swir=True,
        )
    finally:
        batch.convert_image_to_las = original_convert

    assert len(result.failed) == 0
    assert seen_inputs == ["plot1_SWIR.hdr"]


def test_convert_root_folder_sets_plot_offset_delta_from_plot1(tmp_path: Path) -> None:
    root = tmp_path / "root"
    fused_1 = root / "1" / "1_20260521" / "meta" / "ENVI-fused"
    fused_2 = root / "2" / "2_20260521" / "meta" / "ENVI-fused"
    fused_1.mkdir(parents=True)
    fused_2.mkdir(parents=True)

    (fused_1 / "plot1_VNIR.hdr").write_text(
        "ExtraInfo = { PlotEnterLat:51.000000, PlotEnterLong:5.000000 }\n",
        encoding="utf-8",
    )
    (fused_2 / "plot2_VNIR.hdr").write_text(
        "ExtraInfo = { PlotEnterLat:51.100000, PlotEnterLong:5.200000 }\n",
        encoding="utf-8",
    )

    output_root = tmp_path / "out"
    seen_deltas: dict[str, tuple[float, float] | None] = {}

    original_convert = batch.convert_image_to_las
    original_wgs84_to_rd = batch._wgs84_to_rd_new
    try:
        batch._wgs84_to_rd_new = lambda lat, lon: (lon * 1000.0, lat * 1000.0)

        def fake_convert(config: ConversionConfig) -> Path:
            seen_deltas[config.input_path.name] = config.plot_offset_delta_rd
            config.output_path.parent.mkdir(parents=True, exist_ok=True)
            config.output_path.write_text("ok", encoding="utf-8")
            return config.output_path

        batch.convert_image_to_las = fake_convert

        def build_config(input_hdr: Path, output_las: Path) -> ConversionConfig:
            return ConversionConfig(input_path=input_hdr, output_path=output_las)

        result = batch.convert_root_folder(root, output_root, build_config)
    finally:
        batch.convert_image_to_las = original_convert
        batch._wgs84_to_rd_new = original_wgs84_to_rd

    assert len(result.failed) == 0
    assert seen_deltas["plot1_VNIR.hdr"] == (0.0, 0.0)
    assert seen_deltas["plot2_VNIR.hdr"] == (200.0, 100.0)
