# AutoDraw -- DIY Drawing Robot Plotter and Vectorizer

A low-cost drawing robot system that converts images into vector paths for physical reproduction via a custom-built plotter. The system consists of Python vectorization software, a simulation viewer, robot firmware (ESP8266/Arduino), and the plotter hardware. The goal is to build an affordable alternative to commercial plotters capable of drawing images and colored artwork.

> **Note:** Currently this repository contains only the plotter and vectorizer software. The full physical drawing robot hardware and firmware are in development.

## Core Concept

Instead of printing pixels, the system:

1. Takes an image as input
2. Converts it into vector paths
3. Groups shapes by color
4. Outputs paths ready for a drawing robot

## System Architecture

```
Image
  |
  v
Python GUI
  |
  +-- Vectorization
  |
  +-- Simulation (Tkinter + Canvas)
  |
  v
Vector Paths
  |
  v
Plotter Output (SVG / G-code)
```

## Part 1 -- Python Vectorization Software

### Features

- GUI interface (Tkinter)
- Image upload
- Raster to vector conversion
- Color segmentation
- Drawing simulation
- Path export for robot

### Supported Input

**Vector images:** `.svg` (loaded directly as vector paths)

**Raster images:** `.png`, `.jpg`, `.jpeg`, `.bmp` (must be vectorized first)

### Color Vectorization

To preserve colors, the system uses color clustering:

1. **Color reduction** -- The image is simplified using K-Means clustering. Example: 200,000 colors reduced to 6.
2. **Region extraction** -- Each color becomes a mask defining its region.
3. **Contour detection** -- OpenCV (`cv2.findContours()`) detects the outlines of each color region.
4. **Polygon simplification** -- Contours are simplified via `cv2.approxPolyDP()` to reduce points and prevent overly complex robot movements.

Each shape becomes a data structure:

```python
{
  "color": (r, g, b),
  "points": [(x1, y1), (x2, y2), ...]
}
```

### Simulation Viewer

The program uses Tkinter Canvas to simulate drawing. Features: displays vector paths, preserves per-path color, shows robot drawing order, helps debug vectorization output.

### Software Dependencies

```bash
pip install opencv-python numpy pillow svgpathtools
```

| Library | Purpose |
|---------|---------|
| Tkinter | GUI |
| OpenCV | Image processing |
| NumPy | Math operations |
| Pillow | Image loading |
| svgpathtools | SVG parsing |

## Part 2 -- Plotter Output

The vectorizer currently focuses on generating plotter-ready output:

- **Crosshatch rendering** -- `cmyk_crosshatch_plotter.py`
- **Halftone rendering** -- `cmyk_halftone_plotter.py`
- **Scribble rendering** -- `cmyk_scribble_plotter.py`
- **Marker hatch rendering** -- `marker_hatch_plotter.py`

These scripts convert images into stylistic plotter paths optimized for different artistic effects.

## Future Improvements

- **Physical robot build:** ESP8266-based drawing machine with stepper motors and servo pen lift
- **Better vectorization:** Potrace, Bezier curve fitting, adaptive smoothing
- **Path optimization:** TSP-based ordering, segment merging
- **Animated preview:** Step-by-step drawing simulation
- **Direct streaming:** Python to ESP8266 over WiFi for real-time drawing

## Estimated Hardware Cost (planned)

| Component | Approx. Cost |
|-----------|-------------|
| ESP8266 | $4 |
| Stepper motors | $10 |
| Motor drivers | $4 |
| Frame | $10 |
| Belts | $6 |
| Servo | $3 |
| **Total** | **~$35** |

## License

MIT License. See [LICENSE](LICENSE) for details.
