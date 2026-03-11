# 🖊️ DIY Drawing Robot + Vectorizer

A **low-cost drawing robot system** that converts images into vector paths and reproduces them physically using a robot.

The system consists of:

1. **Python vectorization software**
2. **Simulation viewer**
3. **Robot firmware (ESP8266 / Arduino)**
4. **Physical drawing machine**

The goal is to build a **cheap alternative to plotters** capable of drawing images and colored artwork.

---

# 🧠 Core Idea

Instead of printing pixels, the system:

1. Takes an **image**
2. Converts it into **vector paths**
3. Groups shapes by **color**
4. Sends paths to a **drawing robot**

The robot then draws the image **line by line** using pens.

---

# 🏗️ System Architecture

```
Image
  │
  ▼
Python GUI
  │
  ├── Vectorization
  │
  ├── Simulation (Tkinter + Canvas)
  │
  ▼
Vector Paths
  │
  ▼
Robot Commands
  │
  ▼
ESP8266 / Arduino
  │
  ▼
Motors + Pen
  │
  ▼
Drawing on Paper
```

---

# 🖥️ Part 1 — Python Vectorization Software

## Features

The Python program provides:

* GUI interface (Tkinter)
* Image upload
* Raster → vector conversion
* Color segmentation
* Drawing simulation
* Path export for robot

---

## Supported Input

### Vector images

```
.svg
```

These are loaded directly as vector paths.

---

### Raster images

```
.png
.jpg
.jpeg
.bmp
```

These must be **vectorized first**.

---

# 🎨 Color Vectorization

To preserve colors, the system uses **color clustering**.

### Step 1 — Color reduction

The image is simplified using **K-Means clustering**.

Example:

```
Original: 200,000 colors
↓
Clustered: 6 colors
```

---

### Step 2 — Region extraction

Each color becomes a **mask**.

Example:

```
Blue region
Green region
Purple region
```

---

### Step 3 — Contour detection

OpenCV detects the outlines of each color region.

```
cv2.findContours()
```

---

### Step 4 — Polygon simplification

The contours are simplified to reduce points.

```
cv2.approxPolyDP()
```

This prevents overly complex robot movements.

---

### Result

Each shape becomes:

```python
{
  "color": (r,g,b),
  "points": [(x1,y1),(x2,y2)...]
}
```

---

# 🖼️ Simulation Viewer

The program uses **Tkinter Canvas** to simulate drawing.

Features:

* Displays vector paths
* Preserves color
* Shows robot drawing order
* Helps debug vectorization

Example result:

```
purple paths
blue paths
green paths
```

Instead of black outlines.

---

# 📦 Software Dependencies

Install with:

```
pip install opencv-python numpy pillow svgpathtools
```

Libraries used:

| Library      | Purpose          |
| ------------ | ---------------- |
| Tkinter      | GUI              |
| OpenCV       | image processing |
| NumPy        | math             |
| Pillow       | image loading    |
| svgpathtools | SVG parsing      |

---

# 🤖 Part 2 — Robot Hardware

The robot will physically draw the vector paths.

---

## Main Components

Typical configuration:

```
ESP8266 / ESP32
Stepper motors
Motor drivers
Servo motor (pen lift)
Frame
Belts or rails
```

---

### Controller

Options:

```
ESP8266
ESP32
Arduino Mega
```

ESP8266 allows **WiFi communication**.

---

### Motion System

Two axes:

```
X axis
Y axis
```

Possible mechanisms:

| System      | Pros    | Cons    |
| ----------- | ------- | ------- |
| Belts       | cheap   | stretch |
| Lead screws | precise | slower  |
| CoreXY      | fast    | complex |

---

### Pen Lift

A **servo motor** lifts or lowers the pen.

Commands:

```
PEN_DOWN
PEN_UP
```

---

# 🎨 Multi-Color Drawing

The robot cannot hold multiple pens at once.

Instead the program pauses.

Example sequence:

```
Color: Purple
Draw paths
Pause

Insert purple pen
Continue

Color: Blue
Draw paths
Pause
```

---

# 📡 Communication Protocol

Python will send commands to the robot.

Example commands:

```
PEN_UP
MOVE 120 300
PEN_DOWN
DRAW 150 300
DRAW 160 310
PEN_UP
```

These commands translate into **motor movements**.

---

# 🧭 Path Planning

The vectorizer produces **many shapes**.

To optimize drawing speed, we will later add:

### Nearest neighbor path ordering

Instead of:

```
path A
path B
path C
```

We reorder them to reduce travel.

```
closest path first
```

This reduces drawing time significantly.

---

# 📈 Future Improvements

## 1️⃣ Better vectorization

Current system works best for:

* logos
* illustrations
* simple images

Possible upgrades:

```
Potrace
Bezier curve fitting
Adaptive smoothing
```

---

## 2️⃣ Path optimization

Algorithms:

```
TSP (traveling salesman)
Nearest neighbor
Segment merging
```

---

## 3️⃣ Real robot preview

Add animated simulation:

```
draw path step-by-step
```

---

## 4️⃣ SVG export

Allow exporting processed vector files.

```
Export optimized SVG
```

---

## 5️⃣ Direct robot streaming

Instead of saving files:

```
Python → WiFi → ESP8266
```

Real-time drawing.

---

# 💰 Estimated Cost

If built cheaply:

| Component      | Cost |
| -------------- | ---- |
| ESP8266        | $4   |
| Stepper motors | $10  |
| Drivers        | $4   |
| Frame          | $10  |
| Belts          | $6   |
| Servo          | $3   |

Total:

```
≈ $35
```

Much cheaper than commercial plotters.

---

# 🎯 Project Goals

The project aims to create:

* a **cheap drawing robot**
* a **custom vectorizer**
* a **color-aware plotter system**

while learning:

* robotics
* computer vision
* vector graphics
* motion control

---

# 🚀 End Result

User workflow:

```
1 Upload image
2 Vectorize
3 Preview drawing
4 Send to robot
5 Robot draws it
```

Final output:

```
Real drawing on paper
```


