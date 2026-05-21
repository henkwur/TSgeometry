from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable

from .converter import (
    ConversionConfig,
    _extract_plot_point_wgs84,
    _read_envi_metadata,
    _wgs84_to_rd_new,
    convert_image_to_las,
)


@dataclass(slots=True)
class BatchResult:
    converted: list[Path]
    failed: list[tuple[Path, str]]
    cancelled: bool = False


def _natural_key(text: str) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", text)
    key: list[object] = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part.lower())
    return tuple(key)


def _plot_number(plot_id: str) -> int | None:
    match = re.search(r"\d+", plot_id)
    if not match:
        return None
    return int(match.group())


def discover_envi_fused_hdr_files(root_folder: Path) -> list[Path]:
    """Return ENVI header files found under ENVI-fused folders in a root tree."""
    return discover_envi_fused_hdr_files_filtered(root_folder)


def discover_envi_fused_hdr_files_filtered(
    root_folder: Path,
    *,
    include_vnir: bool = True,
    include_swir: bool = True,
) -> list[Path]:
    """Return ENVI-fused hdr files filtered by VNIR/SWIR name markers."""
    if not root_folder.exists() or not root_folder.is_dir():
        return []

    if not include_vnir and not include_swir:
        return []

    def _is_selected(path: Path) -> bool:
        stem = path.stem.lower()
        is_vnir = "vnir" in stem
        is_swir = "swir" in stem
        return (include_vnir and is_vnir) or (include_swir and is_swir)

    files = [
        path
        for path in root_folder.rglob("*.hdr")
        if path.parent.name.lower() == "envi-fused" and _is_selected(path)
    ]
    return sorted(files)


def infer_plot_id(root_folder: Path, input_hdr: Path) -> str:
    """Infer the plot id from first-level folder under root."""
    try:
        rel = input_hdr.relative_to(root_folder)
        if rel.parts:
            return rel.parts[0]
    except ValueError:
        pass
    return input_hdr.parent.name


def convert_root_folder(
    root_folder: Path,
    output_root: Path,
    config_builder: Callable[[Path, Path], ConversionConfig],
    should_cancel: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, Path], None] | None = None,
    include_vnir: bool = True,
    include_swir: bool = True,
) -> BatchResult:
    """Convert all ENVI-fused hdr files under root into per-plot output subfolders."""
    converted: list[Path] = []
    failed: list[tuple[Path, str]] = []
    cancelled = False

    files = discover_envi_fused_hdr_files_filtered(
        root_folder,
        include_vnir=include_vnir,
        include_swir=include_swir,
    )
    plot_ids = {path: infer_plot_id(root_folder, path) for path in files}
    files.sort(
        key=lambda path: (
            _natural_key(plot_ids[path]),
            _natural_key(path.stem),
            str(path).lower(),
        )
    )

    plot_enter_rd_by_file: dict[Path, tuple[float, float]] = {}
    for path in files:
        metadata = _read_envi_metadata(path)
        plot_enter_wgs84 = _extract_plot_point_wgs84(metadata, "PlotEnter")
        if plot_enter_wgs84 is None:
            continue
        plot_enter_rd_by_file[path] = _wgs84_to_rd_new(plot_enter_wgs84[0], plot_enter_wgs84[1])

    reference_plot_enter_rd: tuple[float, float] | None = None
    for path in files:
        if _plot_number(plot_ids[path]) == 1 and path in plot_enter_rd_by_file:
            reference_plot_enter_rd = plot_enter_rd_by_file[path]
            break
    if reference_plot_enter_rd is None:
        for path in files:
            if path in plot_enter_rd_by_file:
                reference_plot_enter_rd = plot_enter_rd_by_file[path]
                break

    total = len(files)
    for index, input_hdr in enumerate(files, start=1):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break

        if on_progress is not None:
            on_progress(index, total, input_hdr)

        plot_id = plot_ids[input_hdr]
        output_folder = output_root / plot_id
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"{input_hdr.stem}.las"

        try:
            config = config_builder(input_hdr, output_path)
            current_plot_enter_rd = plot_enter_rd_by_file.get(input_hdr)
            if reference_plot_enter_rd is not None and current_plot_enter_rd is not None:
                config.plot_offset_reference_plot_enter_rd = reference_plot_enter_rd
                config.plot_offset_delta_rd = (
                    current_plot_enter_rd[0] - reference_plot_enter_rd[0],
                    current_plot_enter_rd[1] - reference_plot_enter_rd[1],
                )
            convert_image_to_las(config)
            converted.append(output_path)
        except Exception as exc:  # noqa: BLE001
            failed.append((input_hdr, str(exc)))

    return BatchResult(converted=converted, failed=failed, cancelled=cancelled)
