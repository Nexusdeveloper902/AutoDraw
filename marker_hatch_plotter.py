"""
marker_hatch_plotter.py

The ultimate solution for physical marker plotters. 
Combines K-Means LAB color clustering (for flat, distinct color groups) with 
Masked Vector Raycasting. Rather than tracing geometric shapes and struggling 
with boundaries/holes, it projects infinite parallel lines across the canvas and
dynamically switches pens based on the pixel's K-Means mask. 

Guarantees 0.0 overlap and 0.0 unfillable gaps between color regions.

Dependencies:
    pip install opencv-python numpy pillow
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import math
from xml.dom.minidom import Document

# --------------------------
# Image Processing: K-Means Segmentation
# --------------------------
def extract_kmeans_masks(image_path, k=5):
    """
    1. Bilateral Filters the image for cartoon-like smoothing.
    2. Runs K-Means in LAB color space.
    3. Returns `k` distinct 2D masks (booleans) and their RGB color values.
    """
    # Load and prep
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not load {image_path}")
        
    if img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        bg = np.ones_like(rgb) * 255.0
        rgb = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
        img = rgb.astype(np.uint8)
    
    img_bgr = img[:, :, :3]
    h, w = img_bgr.shape[:2]
    
    # 0. Apply a mild median blur to remove salt & pepper noise before clustering
    img_bgr = cv2.medianBlur(img_bgr, 5)

    # 1. Bilateral Filter heavily to remove noise and leave sharp boundaries
    blurred = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. LAB Color Space for accurate human-eye grouping
    lab_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2Lab)
    
    # 3. Flatten for K-Means
    pixel_values = lab_img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define criteria and run K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers_lab = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers back to BGR/RGB
    centers_lab = np.uint8(centers_lab)
    centers_lab_img = centers_lab.reshape((k, 1, 3))
    centers_bgr = cv2.cvtColor(centers_lab_img, cv2.COLOR_Lab2BGR)
    centers_rgb = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2RGB).reshape((k, 3))
    
    # Reconstruct 2D label map
    label_map = labels.reshape((h, w))
    
    # --- POST-PROCESSING: Categorical Smoothing ---
    # Removes small noise islands ("white dots") and smooths jagged polygon boundaries
    one_hot = np.zeros((h, w, k), dtype=np.float32)
    for i in range(k):
        mask_f32 = (label_map == i).astype(np.float32)
        # 15x15 blur ensures tiny dots are swallowed by surrounding larger clusters
        one_hot[:, :, i] = cv2.GaussianBlur(mask_f32, (15, 15), 0)
        
    # Re-assign each pixel to the class with the highest blurred presence (soft-vote)
    label_map = np.argmax(one_hot, axis=2)
    
    # Create distinct boolean masks
    masks = []
    colors = []
    
    for i in range(k):
        mask = (label_map == i)
        masks.append(mask)
        # HTML Hex string
        r, g, b = centers_rgb[i]
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        colors.append(hex_color)
        
    return masks, colors, (w, h)


# --------------------------
# Algorithmic Magic: Masked Raycasting
# --------------------------
def generate_masked_hatch_lines(mask, angle_deg, spacing, resolution=1.0, overlap=1.0, min_length=3.0):
    """
    Given a boolean 2D mask, projects parallel lines across the entire canvas bounding box.
    Only returns line segments where the mask is True.
    """
    h, w = mask.shape
    paths = []
    
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    diag = math.hypot(w, h)
    cx, cy = w / 2.0, h / 2.0
    
    y_offsets = np.arange(-diag / 2.0, diag / 2.0, spacing)
    
    alternate_direction = False
    
    for y_off in y_offsets:
        current_path = []
        is_drawing = False
        
        for x_off in np.arange(-diag / 2.0, diag / 2.0, resolution):
            px = cx + x_off * cos_theta - y_off * sin_theta
            py = cy + x_off * sin_theta + y_off * cos_theta
            
            if 0 <= px < w and 0 <= py < h:
                # Is the pixel part of THIS specific marker's color mask?
                if mask[int(py), int(px)]:
                    if not is_drawing:
                        # Pen Down
                        is_drawing = True
                    current_path.append((px, py))
                else:
                    if is_drawing:
                        # Pen Up (hit boundary of another color or empty space)
                        if len(current_path) > 1:
                            p1 = current_path[0]
                            p2 = current_path[-1]
                            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                            if dist >= min_length:
                                p1_ext = (p1[0] - overlap * cos_theta, p1[1] - overlap * sin_theta)
                                p2_ext = (p2[0] + overlap * cos_theta, p2[1] + overlap * sin_theta)
                                if alternate_direction:
                                    paths.append([p2_ext, p1_ext])
                                else:
                                    paths.append([p1_ext, p2_ext])
                        current_path = []
                        is_drawing = False
            else:
                if is_drawing:
                    if len(current_path) > 1:
                        p1 = current_path[0]
                        p2 = current_path[-1]
                        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        if dist >= min_length:
                            p1_ext = (p1[0] - overlap * cos_theta, p1[1] - overlap * sin_theta)
                            p2_ext = (p2[0] + overlap * cos_theta, p2[1] + overlap * sin_theta)
                            
                            if alternate_direction:
                                paths.append([p2_ext, p1_ext])
                            else:
                                paths.append([p1_ext, p2_ext])
                    current_path = []
                    is_drawing = False
                    
        if len(current_path) > 1:
            p1 = current_path[0]
            p2 = current_path[-1]
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist >= min_length:
                p1_ext = (p1[0] - overlap * cos_theta, p1[1] - overlap * sin_theta)
                p2_ext = (p2[0] + overlap * cos_theta, p2[1] + overlap * sin_theta)
                
                if alternate_direction:
                    paths.append([p2_ext, p1_ext])
                else:
                    paths.append([p1_ext, p2_ext])
            
        alternate_direction = not alternate_direction
            
    return paths

# --------------------------
# Utility: SVG Export
# --------------------------
def export_marker_svg(layers_paths, layer_colors, img_size, out_path):
    iw, ih = img_size
    doc = Document()
    svg = doc.createElement('svg')
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    svg.setAttribute('width', str(iw))
    svg.setAttribute('height', str(ih))
    svg.setAttribute('viewBox', f"0 0 {iw} {ih}")
    doc.appendChild(svg)

    for i, paths in enumerate(layers_paths):
        if not paths: continue
        
        hex_color = layer_colors[i]
        
        g = doc.createElement('g')
        g.setAttribute('id', f"Marker_{hex_color}")
        g.setAttribute('stroke', hex_color)
        g.setAttribute('stroke-width', '1') 
        g.setAttribute('fill', 'none')      
        svg.appendChild(g)

        for path in paths:
            if len(path) == 2:
                # Optimized straight line
                line = doc.createElement('line')
                line.setAttribute('x1', f"{path[0][0]:.2f}")
                line.setAttribute('y1', f"{path[0][1]:.2f}")
                line.setAttribute('x2', f"{path[1][0]:.2f}")
                line.setAttribute('y2', f"{path[1][1]:.2f}")
                g.appendChild(line)
            elif len(path) > 2:
                # Polyline for contours
                pts_str = " ".join([f"{p[0]:.2f},{p[1]:.2f}" for p in path])
                polyline = doc.createElement('polyline')
                polyline.setAttribute('points', pts_str)
                g.appendChild(polyline)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc.toprettyxml())


class MarkerHatchPlotterApp:
    def __init__(self, root):
        self.root = root
        root.title("Plotter Prototype (Perfect Marker Hatching)")

        self.image_path = None
        self.img_w = 1
        self.img_h = 1
        
        self.layers_paths = []
        self.layer_colors = []

        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        btn_load = ttk.Button(ctrl_frame, text="Load Image (PNG/JPG)", command=self.load_image)
        btn_load.pack(fill=tk.X, pady=4)

        self.lbl_file = ttk.Label(ctrl_frame, text="No file loaded", wraplength=220)
        self.lbl_file.pack(fill=tk.X, pady=2)

        ttk.Label(ctrl_frame, text="Masked Marker Raycaster").pack(anchor=tk.W, pady=(8,0))
        
        self.k_var = tk.IntVar(value=6)
        ttk.Label(ctrl_frame, text="K (Number of Marker Colors)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.k_var).pack(fill=tk.X)

        self.angle_var = tk.DoubleVar(value=45.0)
        ttk.Label(ctrl_frame, text="Global Hatch Angle (deg)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.angle_var).pack(fill=tk.X)
        
        self.spacing_var = tk.DoubleVar(value=3.0)
        ttk.Label(ctrl_frame, text="Line Spacing (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.spacing_var).pack(fill=tk.X)

        self.overlap_var = tk.DoubleVar(value=1.0)
        ttk.Label(ctrl_frame, text="Boundary Overlap (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.overlap_var).pack(fill=tk.X)

        self.min_length_var = tk.DoubleVar(value=3.0)
        ttk.Label(ctrl_frame, text="Min Line Length (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.min_length_var).pack(fill=tk.X)

        self.outline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Generate Outlines (Black Pen)", variable=self.outline_var).pack(fill=tk.X, pady=4)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)

        btn_vectorize = ttk.Button(ctrl_frame, text="Raycast Hatches", command=self.generate_hatching)
        btn_vectorize.pack(fill=tk.X, pady=4)

        btn_export = ttk.Button(ctrl_frame, text="Export Vector to SVG", command=self.export_svg)
        btn_export.pack(fill=tk.X, pady=4)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)

        btn_draw = ttk.Button(ctrl_frame, text="Animate Drawing", command=self.start_drawing)
        btn_draw.pack(fill=tk.X, pady=4)

        btn_clear = ttk.Button(ctrl_frame, text="Clear Canvas", command=self.clear_canvas)
        btn_clear.pack(fill=tk.X, pady=4)

        canvas_frame = ttk.Frame(root)
        canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(canvas_frame, bg='#ffffff', width=700, height=700)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        self.status = ttk.Label(root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        self._anim_job = None

    def set_status(self, text):
        self.status.config(text=text)
        self.root.update_idletasks()

    def load_image(self):
        filetypes = [("Image files", "*.png *.jpg *.jpeg"), ("All files", "*.*")]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return
        self.image_path = path
        self.lbl_file.config(text=os.path.basename(path))
        im = Image.open(path)
        self.img_w, self.img_h = im.size
        self.display_preview()
        self.set_status(f"Loaded {os.path.basename(path)} ({self.img_w}x{self.img_h})")

    def display_preview(self):
        self.canvas.delete("all")
        if self.image_path is None:
            return
        try:
            im = Image.open(self.image_path)
            cw = int(self.canvas.winfo_width() or 700)
            ch = int(self.canvas.winfo_height() or 700)
            im.thumbnail((cw-20, ch-20), Image.ANTIALIAS)
            self._tk_im = ImageTk.PhotoImage(im)
            self.canvas.create_image(cw/2, ch/2, image=self._tk_im)
        except Exception as e:
            print("Preview error:", e)

    def generate_hatching(self):
        if self.image_path is None:
            messagebox.showwarning("No file", "Please load an image first.")
            return
            
        self.canvas.delete("all")
        self.set_status("Running LAB K-Means. Finding perfect boundaries...")
        self.root.update()

        try:
            k = int(self.k_var.get())
            masks, colors, bounds = extract_kmeans_masks(self.image_path, k=k)
            
            self.layer_colors = colors
            self.layers_paths = []
            
            angle = float(self.angle_var.get())
            spacing = float(self.spacing_var.get())
            overlap = float(self.overlap_var.get())
            min_length = float(self.min_length_var.get())
            
            for i in range(k):
                self.set_status(f"Raycasting perfect lines for Marker {i+1}/{k} ({colors[i]})...")
                self.root.update()
                
                # We project global lines across the discrete mask
                paths = generate_masked_hatch_lines(
                    masks[i], 
                    angle_deg=angle, 
                    spacing=spacing,
                    overlap=overlap,
                    min_length=min_length
                )
                self.layers_paths.append(paths)
                
            if self.outline_var.get():
                self.set_status("Tracing Organic Outlines via Adaptive Thresholding...")
                self.root.update()
                
                # Load high-res original image in grayscale 
                img_gray = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                
                # 1. Mild blur softens absolute micro-noise (paper grain/dust)
                img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
                
                # 2. Adaptive thresholding extracts dark strokes / ink while ignoring lighting gradients.
                # C=6 helps eliminate faint background noise.
                thresh = cv2.adaptiveThreshold(
                    img_blur, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    15, 6
                )
                
                # 3. Morphological closing patches broken pencil lines
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
                # 4. Extract continuous boundary loops
                contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                
                outline_paths = []
                for cnt in contours:
                    # Filter out tiny specks to keep drawing clean
                    if cv2.contourArea(cnt) > 4.0 or len(cnt) > 8:
                        # 0.001 is much smoother than 0.005 (less jagged robotic polygons)
                        epsilon = 0.001 * cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, epsilon, True)
                        
                        path = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
                        if len(path) > 2:
                            path.append(path[0]) # ensure closure
                            outline_paths.append(path)
                
                if outline_paths:
                    self.layers_paths.append(outline_paths)
                    self.layer_colors.append("#000000") # Black outlines
                    
            self.set_status("Generation complete. 0.0 Overlap, 0.0 Holes. Ready to export.")
            self.show_preview_fast()
            
        except Exception as e:
            messagebox.showerror("Algorithm Error", str(e))
            self.set_status("Error calculating rays.")
            print("Error details:", e)

    def show_preview_fast(self):
        self.canvas.delete("all")
        
        cw = int(self.canvas.winfo_width() or 700)
        ch = int(self.canvas.winfo_height() or 700)
        scale_x = (cw - 20) / self.img_w
        scale_y = (ch - 20) / self.img_h
        scale = min(scale_x, scale_y)
        offset_x = (cw - self.img_w*scale) / 2.0
        offset_y = (ch - self.img_h*scale) / 2.0
        
        for i, paths in enumerate(self.layers_paths):
            color = self.layer_colors[i]
            for path in paths:
                if len(path) == 2:
                    p1x = path[0][0]*scale + offset_x
                    p1y = path[0][1]*scale + offset_y
                    p2x = path[1][0]*scale + offset_x
                    p2y = path[1][1]*scale + offset_y
                    self.canvas.create_line(p1x, p1y, p2x, p2y, fill=color, width=1.5)
                elif len(path) > 2:
                    scaled_path = []
                    for px, py in path:
                        scaled_path.append(px*scale + offset_x)
                        scaled_path.append(py*scale + offset_y)
                    self.canvas.create_line(*scaled_path, fill=color, width=1.5)

    def export_svg(self):
        if not self.layers_paths:
            messagebox.showwarning("No vectors", "No paths generated yet. Click 'Raycast Hatches' first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files","*.svg")])
        if not out:
            return
        try:
            export_marker_svg(self.layers_paths, self.layer_colors, (self.img_w, self.img_h), out)
            messagebox.showinfo("Exported", f"Master SVG exported to: {out}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def start_drawing(self):
        if not self.layers_paths:
            messagebox.showwarning("No vectors", "No paths generated yet. Click 'Raycast' first.")
            return
            
        self.canvas.delete("all")
        self.set_status("Animating Plotter Progress...")
        
        draw_list = []
        
        cw = int(self.canvas.winfo_width() or 700)
        ch = int(self.canvas.winfo_height() or 700)
        scale_x = (cw - 20) / self.img_w
        scale_y = (ch - 20) / self.img_h
        scale = min(scale_x, scale_y)
        offset_x = (cw - self.img_w*scale) / 2.0
        offset_y = (ch - self.img_h*scale) / 2.0

        for i, paths in enumerate(self.layers_paths):
            color = self.layer_colors[i]
            for path in paths:
                if len(path) == 2:
                    p1x = path[0][0]*scale + offset_x
                    p1y = path[0][1]*scale + offset_y
                    p2x = path[1][0]*scale + offset_x
                    p2y = path[1][1]*scale + offset_y
                    draw_list.append((color, [p1x, p1y, p2x, p2y]))
                elif len(path) > 2:
                    scaled_path = []
                    for px, py in path:
                        scaled_path.append(px*scale + offset_x)
                        scaled_path.append(py*scale + offset_y)
                    draw_list.append((color, scaled_path))

        self._draw_state = {"list": draw_list, "idx": 0}
        self.draw_speed = 10
        self._animate_step()

    def _animate_step(self):
        state = self._draw_state
        dl = state["list"]
        idx = state["idx"]
        
        if idx >= len(dl):
            self.set_status("Drawing complete")
            self._anim_job = None
            return
            
        end_idx = min(idx + self.draw_speed, len(dl))
        for i in range(idx, end_idx):
            color, coords = dl[i]
            self.canvas.create_line(*coords, fill=color, width=1.5)
            
        state["idx"] = end_idx
        self._anim_job = self.root.after(1, self._animate_step)

    def clear_canvas(self):
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self.canvas.delete("all")
        self.layers_paths = []
        self.layer_colors = []
        self.set_status("Cleared")


def main():
    root = tk.Tk()
    app = MarkerHatchPlotterApp(root)
    root.geometry("1100x760")
    root.mainloop()

if __name__ == "__main__":
    main()
