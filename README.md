# AutoDraw -- DIY Drawing Robot and Vectorizer

A low-cost drawing robot system that converts images into vector paths and reproduces them physically using a custom-built plotter. The system consists of Python vectorization software, a simulation viewer, robot firmware (ESP8266/Arduino), and a physical drawing machine. The goal is to build an affordable alternative to commercial plotters capable of drawing images and colored artwork.

## Core Concept

Instead of printing pixels, the system:

1. Takes an image as input
2. Converts it into vector paths
3. Groups shapes by color
4. Sends paths to a drawing robot

The robot then draws the image line by line using physical pens.

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
Robot Commands
  |
  v
ESP8266 / Arduino
  |
  v
Motors + Pen
  |
  v
Drawing on Paper
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

## Part 2 -- Robot Hardware

The robot physically draws the vector paths.

### Main Components

- ESP8266 / ESP32 controller
- Stepper motors
- Motor drivers
- Servo motor (pen lift)
- Frame and belts or rails

### Motion System

Two axes (X and Y). Possible mechanisms:

| System | Pros | Cons |
|--------|------|------|
| Belts | Cheap | Stretch over time |
| Lead screws | Precise | Slower |
| CoreXY | Fast | More complex build |

### Pen Lift

A servo motor lifts or lowers the pen via `PEN_DOWN` / `PEN_UP` commands.

### Multi-Color Drawing

The robot holds one pen at a time. The program pauses between colors:

```
Color: Purple -- Draw paths -- Pause
Insert purple pen -- Continue
Color: Blue -- Draw paths -- Pause
```

## Communication Protocol

Python sends commands to the robot over serial/WiFi:

```
PEN_UP
MOVE 120 300
PEN_DOWN
DRAW 150 300
DRAW 160 310
PEN_UP
```

These commands translate into motor movements on the controller.

## Path Planning

To optimize drawing speed, the system can reorder paths using nearest-neighbor ordering to reduce travel distance between disconnected shapes.

## Future Improvements

- **Better vectorization:** Potrace, Bezier curve fitting, adaptive smoothing
- **Path optimization:** TSP-based ordering, segment merging
- **Animated preview:** Step-by-step drawing simulation
- **SVG export:** Export processed vector files
- **Direct streaming:** Python to ESP8266 over WiFi for real-time drawing

## Estimated Hardware Cost

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
