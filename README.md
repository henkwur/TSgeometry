# image2las

A small Python CLI for turning image-encoded values into a LAS point cloud.

Each pixel becomes one point. By default the pixel brightness is used as height, but you can also choose a specific RGB channel.

## Setup

Install the project dependencies in a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

## Usage

```bash
image2las input.png output.las
```

### Geometric Accuracy (Millimeter Precision)

For accurate spatial coordinates, specify the physical dimensions and select the appropriate band:

```bash
# Example: ENVI hyperspectral with 2.073m × 2.977m × 0.791m physical dimensions
image2las data.hdr output.las --band 93 \
  --x-scale 0.0020244141 \
  --y-scale 0.0019318624 \
  --z-scale 0.0000706629
```

Scale factors are calculated as:
- `x_scale = physical_width_m / pixel_width`
- `y_scale = physical_height_m / pixel_height`
- `z_scale = physical_height_m / max_value_in_band`

The LAS format ensures 1mm precision through proper scaling of integer coordinates.

### Common Options

```bash
image2las input.png output.las --channel red --x-scale 0.5 --y-scale 0.5 --z-scale 2.0
image2las data.hdr output.las --band 54  # Use specific spectral band for height
```

## File Format Support

- **PNG/JPG**: Grayscale or RGB images; uses `luma`, `red`, `green`, or `blue` channel
- **TIFF**: Multi-band TIFF files; averaged across bands by default or selected with `--band`
- **ENVI (.hdr/.raw)**: Hyperspectral imagery; specify band index with `--band`

## Notes

- Grayscale images are treated as height maps.
- RGB images can be converted using `luma`, `red`, `green`, or `blue`.
- TIFF files with many bands are read through `tifffile` and collapsed to a height map by averaging the bands.
- ENVI hyperspectral files are read through the `spectral` library.
- Transparent pixels are skipped when the image has an alpha channel.
- The default band ordering for many hyperspectral sensors is `[NIR, Red, Green]`; adjust band indices accordingly.
