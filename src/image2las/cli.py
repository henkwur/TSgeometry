from __future__ import annotations

import argparse
from pathlib import Path

from .batch import convert_root_folder
from .converter import ConversionConfig, convert_image_to_las


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image2las",
        description="Convert image-encoded data into a LAS point cloud.",
    )
    parser.add_argument("input", type=Path, help="Path to the source image file")
    parser.add_argument("output", type=Path, help="Path to the output LAS file")
    parser.add_argument(
        "--channel",
        choices=("luma", "red", "green", "blue"),
        default="luma",
        help="Image channel used as height information",
    )
    parser.add_argument(
        "--band",
        type=int,
        default=None,
        help="Specific band index for hyperspectral images (overrides --channel)",
    )
    parser.add_argument(
        "--envi-coordinates",
        action="store_true",
        help="Extract XYZ coordinates from ENVI encoded meter/fraction channel pairs",
    )
    parser.add_argument("--x-meter-channel", type=int, default=227, help="1-based ENVI channel for X whole meters")
    parser.add_argument("--x-fraction-channel", type=int, default=228, help="1-based ENVI channel for X fraction (0.1mm)")
    parser.add_argument("--y-meter-channel", type=int, default=229, help="1-based ENVI channel for Y whole meters")
    parser.add_argument("--y-fraction-channel", type=int, default=230, help="1-based ENVI channel for Y fraction (0.1mm)")
    parser.add_argument("--z-meter-channel", type=int, default=231, help="1-based ENVI channel for Z whole meters")
    parser.add_argument("--z-fraction-channel", type=int, default=232, help="1-based ENVI channel for Z fraction (0.1mm)")
    parser.add_argument("--no-rgb", action="store_true", help="Disable RGB color output in LAS")
    parser.add_argument("--red-channel", type=int, default=93, help="1-based channel used for red color in hyperspectral input")
    parser.add_argument("--green-channel", type=int, default=54, help="1-based channel used for green color in hyperspectral input")
    parser.add_argument("--blue-channel", type=int, default=24, help="1-based channel used for blue color in hyperspectral input")
    parser.add_argument("--rgb-clip-low", type=float, default=1.0, help="Lower RGB percentile used for contrast stretch")
    parser.add_argument("--rgb-clip-high", type=float, default=99.5, help="Upper RGB percentile used for contrast stretch")
    parser.add_argument("--x-scale", type=float, default=1.0, help="Spacing between columns")
    parser.add_argument("--y-scale", type=float, default=1.0, help="Spacing between rows")
    parser.add_argument("--z-scale", type=float, default=1.0, help="Height multiplier")
    parser.add_argument("--x-offset", type=float, default=0.0, help="X origin offset")
    parser.add_argument("--y-offset", type=float, default=0.0, help="Y origin offset")
    parser.add_argument("--z-offset", type=float, default=0.0, help="Z origin offset")
    parser.add_argument(
        "--no-invert-y",
        action="store_true",
        help="Keep image row order instead of flipping the vertical axis",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    def _build_config(input_path: Path, output_path: Path) -> ConversionConfig:
        return ConversionConfig(
            input_path=input_path,
            output_path=output_path,
            channel=args.channel,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            z_scale=args.z_scale,
            x_offset=args.x_offset,
            y_offset=args.y_offset,
            z_offset=args.z_offset,
            invert_y=not args.no_invert_y,
            band_index=args.band,
            use_envi_coordinates=args.envi_coordinates,
            x_meter_channel=args.x_meter_channel,
            x_fraction_channel=args.x_fraction_channel,
            y_meter_channel=args.y_meter_channel,
            y_fraction_channel=args.y_fraction_channel,
            z_meter_channel=args.z_meter_channel,
            z_fraction_channel=args.z_fraction_channel,
            use_rgb_colors=not args.no_rgb,
            red_channel=args.red_channel,
            green_channel=args.green_channel,
            blue_channel=args.blue_channel,
            rgb_clip_low_percentile=args.rgb_clip_low,
            rgb_clip_high_percentile=args.rgb_clip_high,
        )

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        result = convert_root_folder(args.input, args.output, _build_config)
        for output_path in result.converted:
            print(f"Converted: {output_path}")
        for input_path, message in result.failed:
            print(f"Failed: {input_path} -> {message}")
        if result.failed:
            return 1
        if not result.converted:
            print("No ENVI-fused .hdr files found under input root.")
            return 1
        return 0

    config = _build_config(args.input, args.output)
    convert_image_to_las(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
