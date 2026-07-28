# AutoDraw -- Image to Plotter Vectorizer

A Python-based vectorization toolkit that converts raster images into plotter-ready vector paths using multiple rendering techniques. Built with OpenCV, NumPy, and Tkinter.

## Overview

AutoDraw takes an image and produces plotter-optimized vector output using several distinct rendering engines. Each engine approaches the image-to-path problem differently, producing different artistic styles suitable for pen plotters, laser engravers, and other CNC drawing machines.

## Rendering Engines

### CMYK Crosshatch Renderer (`cmyk_crosshatch_plotter.py`)

Converts images to CMYK channels and generates angled hatch lines for each color layer. Uses intensity thresholds along rotated scan lines to create directional shading patterns. Outputs SVG with per-layer color separation.

### CMYK Halftone Renderer (`cmyk_halftone_plotter.py`)

Produces continuous amplitude-modulated sine waves along rotated structural axes for each CMYK layer. Creates halftone-like optical illusions without geometric clustering, eliminating mechanical gaps and double-drawing artifacts.

### CMYK Scribble Renderer (`cmyk_scribble_plotter.py`)

Simulates an artist scribbling with CMYK pens using density-driven random walks. Drops a virtual pen at the darkest local pixel, draws toward it, and subtracts that darkness from the canvas to force exploration of new areas.

### Marker Hatch Renderer (`marker_hatch_plotter.py`)

Combines K-Means LAB color clustering with masked vector raycasting. Projects infinite parallel lines across the canvas and dynamically assigns pen colors based on pixel masks, guaranteeing zero overlap and zero unfillable gaps between color regions.

### Prototype Vectorizer (`test.py`)

The original reference implementation featuring K-Means color clustering, contour detection via OpenCV, polygon simplification with Ramer-Douglas-Peucker, SVG export, and an animated drawing simulator.

## Color Processing Pipeline

All renderers share a common color separation approach:

1. **Color separation** -- RGB image is converted to CMYK (Cyan, Magenta, Yellow, Black) channels
2. **Gamma correction** -- A gamma curve is applied to cut out background noise and compression artifacts
3. **Per-layer rendering** -- Each channel is processed independently with its specific rendering algorithm
4. **SVG output** -- Final paths are combined into a color-separated SVG file

## Installation

```bash
pip install opencv-python numpy pillow svgpathtools
```

| Library | Purpose |
|---------|---------|
| Tkinter | GUI interface |
| OpenCV | Image processing, contour detection |
| NumPy | Array math, vector operations |
| Pillow | Image loading |
| svgpathtools | SVG parsing (prototype) |

## Usage

Each renderer is a standalone script with a Tkinter GUI:

```bash
python cmyk_crosshatch_plotter.py
python cmyk_halftone_plotter.py
python cmyk_scribble_plotter.py
python marker_hatch_plotter.py
python test.py
```

Each GUI provides:
- Image upload (PNG, JPG, BMP)
- Renderer-specific parameter controls (spacing, angle, threshold, etc.)
- Live preview canvas
- SVG export

## Scope

This repository contains only the vectorizer software. There is no hardware, firmware, or physical drawing machine component included. The SVG output is compatible with any standard pen plotter, laser engraver, or CNC machine that accepts SVG or can have SVG converted to its native format.

## License

MIT License. See [LICENSE](LICENSE) for details.
