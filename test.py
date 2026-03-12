"""
plotter_prototype_color.py

Tkinter GUI that:
 - accepts PNG/JPG/SVG uploads
 - color-vectorizes PNG/JPG -> k-means color regions -> contours -> simplified polygons
 - parses SVG -> polylines (by sampling path segments)
 - previews colored regions on a canvas
 - animates the drawing (simulator) with color support
 - exports the vectorized result to an SVG with fills

Dependencies:
    pip install opencv-python numpy pillow svgpathtools
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import math
from svgpathtools import svg2paths2
from xml.dom.minidom import Document

# --------------------------
# Utility: simplify polyline (Ramer-Douglas-Peucker)
# --------------------------
def simplify_polyline(points, tol=2.0):
    """
    Ramer-Douglas-Peucker polyline simplification
    points: list of (x,y)
    tol: distance tolerance
    """
    if len(points) < 3:
        return points

    def point_line_distance(pt, a, b):
        ax, ay = a; bx, by = b; px, py = pt
        dx = bx - ax; dy = by - ay
        if dx == 0 and dy == 0:
            return math.hypot(px-ax, py-ay)
        t = ((px-ax)*dx + (py-ay)*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        projx = ax + t*dx; projy = ay + t*dy
        return math.hypot(px-projx, py-projy)

    def rdp(pts):
        if len(pts) < 3:
            return pts
        a = pts[0]; b = pts[-1]
        max_d = -1; index = -1
        for i in range(1, len(pts)-1):
            d = point_line_distance(pts[i], a, b)
            if d > max_d:
                max_d = d; index = i
        if max_d <= tol:
            return [a, b]
        left = rdp(pts[:index+1])
        right = rdp(pts[index:])
        return left[:-1] + right

    return rdp(points)

# --------------------------
# Utility: Polygon Hatching (Scan-line)
# --------------------------
def hatch_polygon(boundaries, angle_deg=45.0, spacing=4.0):
    """
    Fills a polygon (with optional holes) with parallel hatch lines.
    boundaries: list of point lists. boundaries[0] is exterior, rest are holes.
    Returns: list of polylines (each is a list of 2 points: [start, end])
    """
    if not boundaries or len(boundaries[0]) < 3:
        return []

    # Convert angle to radians
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # 1. Rotate polygon vertices so hatch lines become horizontal (y = constant)
    rotated_boundaries = []
    min_y = float('inf')
    max_y = float('-inf')

    for boundary in boundaries:
        if len(boundary) < 3: continue
        rotated = []
        for x, y in boundary:
            rx = x * cos_theta + y * sin_theta
            ry = -x * sin_theta + y * cos_theta
            rotated.append((rx, ry))
            if ry < min_y: min_y = ry
            if ry > max_y: max_y = ry
        rotated_boundaries.append(rotated)

    # 2. Generate scan lines and find intersections across ALL rings (even-odd rule)
    lines = []
    # Start slightly inside to avoid edge precision issues
    y = min_y + (spacing / 2.0)
    
    while y <= max_y:
        intersections = []
        # Check intersection of horizontal line 'y' with each polygon edge
        for rotated in rotated_boundaries:
            for i in range(len(rotated)):
                p1 = rotated[i]
                p2 = rotated[(i + 1) % len(rotated)]
                
                # Check if edge crosses the scan line y
                if (p1[1] <= y < p2[1]) or (p2[1] <= y < p1[1]):
                    # Calculate x intersection using linear interpolation
                    if p2[1] != p1[1]: # Avoid division by zero
                        x_int = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                        intersections.append(x_int)
        
        # Sort intersections left to right
        intersections.sort()
        
        # Pair up intersections to form line segments.
        # This implicitly skips holes since it switches 'inside'/'outside' state at each boundary.
        for i in range(0, len(intersections) - 1, 2):
            x1 = intersections[i]
            x2 = intersections[i+1]
            
            # Rotate points back to original orientation
            start_x = x1 * cos_theta - y * sin_theta
            start_y = x1 * sin_theta + y * cos_theta
            
            end_x = x2 * cos_theta - y * sin_theta
            end_y = x2 * sin_theta + y * cos_theta
            
            lines.append([(start_x, start_y), (end_x, end_y)])
            
        y += spacing

    # 3. Connect adjacent hatch lines into a continuous zig-zag path to minimize plotter lifts
    if not lines:
        return []

    # Simple alternating connection for zig-zag
    # Reverse every other line
    zigzag_path = []
    for i, line in enumerate(lines):
        if i % 2 == 1:
            zigzag_path.extend([line[1], line[0]])
        else:
            zigzag_path.extend([line[0], line[1]])
            
    # Return as a single polyline (or multiple if we wanted to break long jumps)
    return [zigzag_path] if zigzag_path else []

# --------------------------
# Utility: SVG parsing -> polylines (unchanged)
# --------------------------
def sample_svg_paths(svg_path, samples_per_segment=20):
    """
    Read SVG file and convert to a list of polylines.
    Each polyline is a list of (x, y) tuples in the SVG coordinate space.
    Returns list of dicts: {"color": None, "points": [(x,y),...]}
    """
    paths, attributes, svg_att = svg2paths2(svg_path)
    polylines = []
    for path in paths:
        pts = []
        for seg in path:
            for t in np.linspace(0, 1, samples_per_segment, endpoint=False):
                p = seg.point(t)
                pts.append((p.real, p.imag))
        if len(path) > 0:
            last = path[-1].end
            pts.append((last.real, last.imag))
        if pts:
            simplified = simplify_polyline(pts, tol=0.5)
            if len(simplified) >= 2:
                polylines.append({"color": None, "points": simplified})
    return polylines

# --------------------------
# Utility: Raster color vectorization (k-means -> region masks -> contours)
# --------------------------
def raster_to_color_polylines(image_path, num_colors=12, approx_epsilon=1.0, min_area=50):
    """
    Convert raster image into color regions and vectorize each region.
    Returns list of dicts: {"color": (r,g,b), "points": [(x,y), ...]} and image size (w,h)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Unable to open image: " + image_path)

    # If image has alpha, composite on white background
    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        bg = np.ones_like(rgb) * 255.0
        rgb = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
        img = rgb.astype(np.uint8)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Pre-processing: Bilateral filter to smooth gradients but enforce edge limits
    img_filtered = cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Pre-processing: Move to perceptual LAB space for better K-means grouping
    img_lab = cv2.cvtColor(img_filtered, cv2.COLOR_RGB2LAB)
    h, w = img_lab.shape[:2]

    # reshape for kmeans
    data = img_lab.reshape((-1, 3)).astype(np.float32)

    # K-means clustering
    K = max(1, int(num_colors))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    attempts = 4
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(data, K, None, criteria, attempts, flags)

    # Convert LAB centers back to RGB for display & SVG logic
    centers_lab = np.uint8([centers])
    centers_rgb = cv2.cvtColor(centers_lab, cv2.COLOR_LAB2RGB)[0]

    labels = labels.flatten()
    labels_map = labels.reshape((h, w))

    polylines = []

    # Process each cluster (color)
    for i in range(K):
        color = tuple(int(c) for c in centers_rgb[i])
        mask = (labels_map == i).astype(np.uint8) * 255

        # optionally blur or morphology to remove tiny specks
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # RETR_CCOMP retrieves all the contours and organizes them into a two-level hierarchy
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is None:
            continue
            
        hierarchy = hierarchy[0]

        for idx, cnt in enumerate(contours):
            # Check if this is an external contour (Parent == -1)
            if hierarchy[idx][3] == -1:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                    
                approx = cv2.approxPolyDP(cnt.astype(np.float32), approx_epsilon, True)
                approx = approx.squeeze(axis=1)
                if approx.ndim != 2 or approx.shape[0] < 3:
                    continue
                    
                pts = [tuple(map(float, p)) for p in approx]
                pts = simplify_polyline(pts, tol=1.0)
                
                if len(pts) >= 3:
                    # Find inside holes
                    holes = []
                    child_idx = hierarchy[idx][2]
                    while child_idx != -1:
                        hole_cnt = contours[child_idx]
                        if cv2.contourArea(hole_cnt) >= min_area * 0.2: # Holes can be slightly smaller
                            hole_cnt = hole_cnt.squeeze(axis=1)
                            if hole_cnt.ndim == 2 and hole_cnt.shape[0] >= 3:
                                hole_approx = cv2.approxPolyDP(hole_cnt.astype(np.float32), approx_epsilon, True)
                                hole_approx = hole_approx.squeeze(axis=1)
                                if hole_approx.ndim == 2 and hole_approx.shape[0] >= 3:
                                    hole_pts = [tuple(map(float, p)) for p in hole_approx]
                                    hole_pts = simplify_polyline(hole_pts, tol=1.0)
                                    if len(hole_pts) >= 3:
                                        holes.append(hole_pts)
                        child_idx = hierarchy[child_idx][0]
                        
                    polylines.append({"color": color, "points": pts, "holes": holes, "type": "fill"})

    # sort by area (optional) - larger regions first
    polylines.sort(key=lambda p: -abs(cv2.contourArea(np.array(p["points"], dtype=np.float32)) if len(p["points"])>=3 else 0))
    return polylines, (w, h)

# --------------------------
# Utility: mapping polylines to canvas coords
# --------------------------
def map_polylines_to_canvas(polylines, img_size, canvas_size, padding=10):
    iw, ih = img_size
    cw, ch = canvas_size
    scale_x = (cw - 2*padding) / iw
    scale_y = (ch - 2*padding) / ih
    scale = min(scale_x, scale_y)
    offset_x = (cw - iw*scale) / 2.0
    offset_y = (ch - ih*scale) / 2.0

    mapped = []
    for p in polylines:
        pts = p["points"]
        mapped_pts = [(pt[0]*scale + offset_x, pt[1]*scale + offset_y) for pt in pts]
        
        # Keep track of properties (like 'type' for hatched vs solid)
        mapped_p = {"color": p.get("color", None), "points": mapped_pts}
        if "type" in p:
            mapped_p["type"] = p["type"]
        if "holes" in p:
            mapped_holes = []
            for hole in p["holes"]:
                mapped_holes.append([(hx*scale + offset_x, hy*scale + offset_y) for hx, hy in hole])
            mapped_p["holes"] = mapped_holes
        mapped.append(mapped_p)
    return mapped

# --------------------------
# Utility: export color polylines to SVG
# --------------------------
def export_color_polylines_to_svg(polylines, img_size, out_path, angle_deg=45.0, spacing=4.0):
    iw, ih = img_size
    doc = Document()
    svg = doc.createElement('svg')
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    svg.setAttribute('width', str(iw))
    svg.setAttribute('height', str(ih))
    svg.setAttribute('viewBox', f"0 0 {iw} {ih}")
    doc.appendChild(svg)

    # Group polylines by color to make it plotter friendly
    color_groups = {}
    for region in polylines:
        color = region.get("color", None)
        color_key = "Black" if color is None else "#%02x%02x%02x" % color
        
        if color_key not in color_groups:
            color_groups[color_key] = []
        color_groups[color_key].append(region)

    # For each color group, create a <g> tag
    for color_key, regions in color_groups.items():
        g = doc.createElement('g')
        g.setAttribute('id', f"Pen_{color_key.replace('#', '')}")
        g.setAttribute('stroke', color_key)
        g.setAttribute('stroke-width', '1') # Plotter pen width, conceptual
        g.setAttribute('fill', 'none')      # CRITICAL: Plotters don't fill
        svg.appendChild(g)

        for region in regions:
            pts = region["points"]
            if len(pts) < 2:
                continue
                
            region_type = region.get("type", "stroke")

            if region_type == "fill" and len(pts) >= 3:
                # 1. Draw the outline
                poly = doc.createElement('polygon')
                points_attr = " ".join(f"{p[0]},{p[1]}" for p in pts)
                poly.setAttribute('points', points_attr)
                g.appendChild(poly)
                
                # Draw holes
                for hole in region.get("holes", []):
                    h_poly = doc.createElement('polygon')
                    h_attr = " ".join(f"{p[0]},{p[1]}" for p in hole)
                    h_poly.setAttribute('points', h_attr)
                    g.appendChild(h_poly)
                
                # 2. Add the hatching path
                boundaries = [pts] + region.get("holes", [])
                hatch_paths = hatch_polygon(boundaries, angle_deg, spacing)
                for path in hatch_paths:
                    if len(path) >= 2:
                        hatch_line = doc.createElement('polyline')
                        hatch_attr = " ".join(f"{p[0]},{p[1]}" for p in path)
                        hatch_line.setAttribute('points', hatch_attr)
                        g.appendChild(hatch_line)
            else:
                # It's already just a stroke/polyline (like from loaded SVG)
                poly = doc.createElement('polyline')
                points_attr = " ".join(f"{p[0]},{p[1]}" for p in pts)
                poly.setAttribute('points', points_attr)
                if math.hypot(pts[0][0]-pts[-1][0], pts[0][1]-pts[-1][1]) < 1.0:
                    poly = doc.createElement('polygon') # Close it if endpoints touch
                    poly.setAttribute('points', points_attr)
                g.appendChild(poly)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc.toprettyxml())

# --------------------------
# GUI Application
# --------------------------
class PlotterPrototypeApp:
    def __init__(self, root):
        self.root = root
        root.title("Plotter Prototype (Color Vectorization)")

        self.image_path = None
        self.color_polylines = []   # list of {"color":(r,g,b),"points":[(x,y),...]}
        self.img_w = 1
        self.img_h = 1

        # UI layout
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        btn_load = ttk.Button(ctrl_frame, text="Load Image (PNG/JPG/SVG)", command=self.load_image)
        btn_load.pack(fill=tk.X, pady=4)

        self.lbl_file = ttk.Label(ctrl_frame, text="No file loaded", wraplength=220)
        self.lbl_file.pack(fill=tk.X, pady=2)

        # Color vectorization parameters
        ttk.Label(ctrl_frame, text="Color vectorization (raster)").pack(anchor=tk.W, pady=(8,0))
        self.num_colors_var = tk.IntVar(value=12)
        ttk.Label(ctrl_frame, text="Number of colors (k)").pack(anchor=tk.W)
        ttk.Scale(ctrl_frame, from_=2, to=12, variable=self.num_colors_var, orient=tk.HORIZONTAL).pack(fill=tk.X)

        self.approx_var = tk.DoubleVar(value=1.0)
        ttk.Label(ctrl_frame, text="Approx epsilon (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.approx_var).pack(fill=tk.X)

        self.min_area_var = tk.IntVar(value=50)
        ttk.Label(ctrl_frame, text="Min region area (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.min_area_var).pack(fill=tk.X)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)
        
        ttk.Label(ctrl_frame, text="Plotter Hatching (Fills)").pack(anchor=tk.W)
        self.hatch_spacing_var = tk.DoubleVar(value=4.0)
        ttk.Label(ctrl_frame, text="Line Spacing (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.hatch_spacing_var).pack(fill=tk.X)
        
        self.hatch_angle_var = tk.DoubleVar(value=45.0)
        ttk.Label(ctrl_frame, text="Line Angle (deg)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.hatch_angle_var).pack(fill=tk.X)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)

        btn_vectorize = ttk.Button(ctrl_frame, text="Vectorize / Parse", command=self.vectorize_current)
        btn_vectorize.pack(fill=tk.X, pady=4)

        btn_export = ttk.Button(ctrl_frame, text="Export Vector to SVG", command=self.export_svg)
        btn_export.pack(fill=tk.X, pady=4)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)

        btn_draw = ttk.Button(ctrl_frame, text="Animate Drawing", command=self.start_drawing)
        btn_draw.pack(fill=tk.X, pady=4)

        btn_clear = ttk.Button(ctrl_frame, text="Clear Canvas", command=self.clear_canvas)
        btn_clear.pack(fill=tk.X, pady=4)

        # Canvas for preview/drawing
        canvas_frame = ttk.Frame(root)
        canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(canvas_frame, bg='white', width=700, height=700)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # status bar
        self.status = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        # animation state
        self._anim_job = None

    def set_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    def load_image(self):
        filetypes = [("Image files", "*.png *.jpg *.jpeg *.svg"), ("All files", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return
        self.image_path = path
        self.lbl_file.config(text=os.path.basename(path))
        if path.lower().endswith('.svg'):
            # fallback image size; svg parsing will generate polylines in svg coords
            self.img_w, self.img_h = 1000, 1000
        else:
            im = Image.open(path)
            self.img_w, self.img_h = im.size
        self.display_preview()
        self.set_status(f"Loaded {os.path.basename(path)} ({self.img_w}x{self.img_h})")

    def display_preview(self):
        self.canvas.delete("all")
        if self.image_path is None:
            return
        try:
            if self.image_path.lower().endswith('.svg'):
                self.canvas.create_text(350, 350, text="SVG loaded — click 'Vectorize / Parse' to view paths", font=("Arial", 14), fill="black")
                return
            im = Image.open(self.image_path)
            cw = int(self.canvas.winfo_width() or 700)
            ch = int(self.canvas.winfo_height() or 700)
            im.thumbnail((cw-20, ch-20), Image.ANTIALIAS)
            self._tk_im = ImageTk.PhotoImage(im)
            self.canvas.create_image(cw/2, ch/2, image=self._tk_im)
        except Exception as e:
            print("Preview error:", e)

    def vectorize_current(self):
        if self.image_path is None:
            messagebox.showwarning("No file", "Please load an image first.")
            return
        self.canvas.delete("all")
        self.color_polylines = []
        try:
            if self.image_path.lower().endswith('.svg'):
                self.set_status("Parsing SVG...")
                parsed = sample_svg_paths(self.image_path, samples_per_segment=16)
                # convert parsed polylines to color_polylines format (no color)
                self.color_polylines = [{"color": None, "points": p["points"]} for p in parsed]
                # attempt to set image size from svg (we used fallback earlier)
                # leave img_w,img_h as-is (1000x1000 by default)
            else:
                self.set_status("Color-vectorizing raster image (this may take a few seconds)...")
                k = int(self.num_colors_var.get())
                eps = float(self.approx_var.get())
                min_area = int(self.min_area_var.get())
                polylines, (w, h) = raster_to_color_polylines(self.image_path, num_colors=k, approx_epsilon=eps, min_area=min_area)
                self.color_polylines = polylines
                self.img_w, self.img_h = w, h
            self.show_polylines()
            self.set_status(f"Vectorization done — {len(self.color_polylines)} color regions")
        except Exception as e:
            messagebox.showerror("Vectorization error", str(e))
            self.set_status("Error during vectorization")

    def show_polylines(self):
        self.canvas.delete("all")
        if not self.color_polylines:
            return
        cw = int(self.canvas.winfo_width() or 700)
        ch = int(self.canvas.winfo_height() or 700)
        mapped = map_polylines_to_canvas(self.color_polylines, (self.img_w, self.img_h), (cw, ch), padding=10)
        self._mapped_polylines = mapped

        for region in mapped:
            pts = region["points"]
            if len(pts) < 3:
                # draw as polyline
                coords = []
                for x, y in pts:
                    coords.extend((x, y))
                if len(coords) >= 4:
                    self.canvas.create_line(coords, fill="black", width=1.0)
                continue
            # choose color (if None -> gray outline)
            color = region.get("color", None)
            if color is None:
                fill = ""
                outline = "black"
            else:
                r, g, b = color
                fill = "#%02x%02x%02x" % (r, g, b)
                outline = ""  # no outline for filled regions
            # detect if shape is closed (first ~ last)
            first = pts[0]; last = pts[-1]
            closed = math.hypot(first[0]-last[0], first[1]-last[1]) < 8.0
            coords = []
            for x, y in pts:
                coords.extend((x, y))
            if closed:
                # draw polygon outline
                self.canvas.create_polygon(coords, fill="", outline=stroke if stroke else "black", width=1.5)
                
                # draw holes outlines
                for hole in region.get("holes", []):
                    h_coords = []
                    for hx, hy in hole: h_coords.extend((hx, hy))
                    if len(h_coords) >= 6:
                        self.canvas.create_polygon(h_coords, fill="", outline=stroke if stroke else "black", width=1.0)
                
                # optionally show hatching preview here based on settings
                try:
                    spacing = float(self.hatch_spacing_var.get())
                    angle = float(self.hatch_angle_var.get())
                    boundaries = [pts] + region.get("holes", [])
                    hatch_lines = hatch_polygon(boundaries, angle_deg=angle, spacing=spacing)
                    for h_path in hatch_lines:
                        h_coords = []
                        for hx, hy in h_path: h_coords.extend((hx, hy))
                        self.canvas.create_line(h_coords, fill=stroke, width=1.0)
                except ValueError:
                    pass
            else:
                # open shape -> draw stroke in color (or black)
                stroke = fill if fill else "black"
                self.canvas.create_line(coords, fill=stroke, width=1.2)

    def clear_canvas(self):
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self.canvas.delete("all")
        self.color_polylines = []
        self.set_status("Cleared")

    def start_drawing(self):
        if not hasattr(self, "_mapped_polylines") or not self._mapped_polylines:
            messagebox.showwarning("No vectors", "No vector paths to draw. Vectorize or load an SVG first.")
            return
        self.canvas.delete("all")
        self.set_status("Animating drawing...")
        # flatten mapped polylines into per-segment drawable paths (each with color)
        draw_list = []
        for region in self._mapped_polylines:
            pts = region["points"]
            color = region.get("color", None)
            # represent as series of segments along boundary (closed or open)
            if region.get("type") == "fill" and len(pts) >= 3:
                # Animate the outline
                segs = []
                for i in range(len(pts)-1):
                    segs.append((pts[i], pts[i+1]))
                segs.append((pts[-1], pts[0]))
                draw_list.append({"color": color, "segs": segs})
                
                # Animate holes outlines
                for hole in region.get("holes", []):
                    if len(hole) < 3: continue
                    for i in range(len(hole)-1):
                        segs.append((hole[i], hole[i+1]))
                    segs.append((hole[-1], hole[0]))
                
                # Animate the hatching
                try:
                    spacing = float(self.hatch_spacing_var.get())
                    angle = float(self.hatch_angle_var.get())
                    boundaries = [pts] + region.get("holes", [])
                    hatch_paths = hatch_polygon(boundaries, angle_deg=angle, spacing=spacing)
                    for path in hatch_paths:
                        if len(path) < 2: continue
                        h_segs = []
                        for i in range(len(path)-1):
                            h_segs.append((path[i], path[i+1]))
                        draw_list.append({"color": color, "segs": h_segs})
                except ValueError:
                    pass
            else:
                # Standard polyline
                if len(pts) < 2:
                    continue
                segs = []
                for i in range(len(pts)-1):
                    segs.append((pts[i], pts[i+1]))
                # if closed, add segment closing the loop
                if math.hypot(pts[0][0]-pts[-1][0], pts[0][1]-pts[-1][1]) < 8.0:
                    segs.append((pts[-1], pts[0]))
                draw_list.append({"color": color, "segs": segs})

        # animation parameters
        self._draw_state = {"draw_list": draw_list, "region_i": 0, "seg_i": 0, "pos": None}
        self.draw_speed = 4.0  # pixels per frame
        self._animate_step()

    def _animate_step(self):
        state = self._draw_state
        dl = state["draw_list"]
        if state["region_i"] >= len(dl):
            self.set_status("Drawing complete")
            self._anim_job = None
            return
        region = dl[state["region_i"]]
        if state["seg_i"] >= len(region["segs"]):
            # move to next region
            state["region_i"] += 1
            state["seg_i"] = 0
            state["pos"] = None
            self._anim_job = self.root.after(200, self._animate_step)
            return

        a, b = region["segs"][state["seg_i"]]
        if state["pos"] is None:
            state["pos"] = a

        ax, ay = state["pos"]
        bx, by = b
        dx = bx - ax; dy = by - ay
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            state["seg_i"] += 1
            self._anim_job = self.root.after(1, self._animate_step)
            return

        step = min(self.draw_speed, dist)
        nx = ax + dx / dist * step
        ny = ay + dy / dist * step

        # choose color for drawing stroke
        color = region["color"]
        if color is None:
            stroke = "black"
        else:
            stroke = "#%02x%02x%02x" % color

        self.canvas.create_line(ax, ay, nx, ny, fill=stroke, width=2.0)
        state["pos"] = (nx, ny)

        # if reached end of segment
        if math.hypot(bx - nx, by - ny) < 1e-2:
            state["seg_i"] += 1
            state["pos"] = None

        self._anim_job = self.root.after(10, self._animate_step)

    def export_svg(self):
        if not self.color_polylines:
            messagebox.showwarning("No vectors", "No vector paths to export. Vectorize or load an SVG first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files","*.svg")])
        if not out:
            return
        try:
            ang = float(self.hatch_angle_var.get())
            spc = float(self.hatch_spacing_var.get())
            export_color_polylines_to_svg(self.color_polylines, (self.img_w, self.img_h), out, angle_deg=ang, spacing=spc)
            messagebox.showinfo("Exported", f"SVG exported to: {out}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

# --------------------------
# Run the app
# --------------------------
def main():
    root = tk.Tk()
    app = PlotterPrototypeApp(root)
    root.geometry("1100x760")
    root.mainloop()

if __name__ == "__main__":
    main()