"""
cmyk_halftone_plotter.py

A plotter-friendly vectorization engine that uses continuous amplitude-modulated
sine waves along rotated structural axes (CMYK layers) rather than geometric
K-Means clustering. This perfectly eliminates mechanical "holes" or double-drawing 
while producing stunning optical illusions.

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
# Algorithmic Magic: CMYK Amplitude Modulation
# --------------------------

def rgb_to_cmyk_channels(img_bgr):
    """
    Convert a BGR numpy array to 4 separate CMYK intensity maps.
    Returns: dict of 2D numpy arrays {"C", "M", "Y", "K"}
    Values are 0.0 (no ink) to 1.0 (full ink)
    """
    bgr_prime = img_bgr.astype(np.float32) / 255.0
    
    # Extract channels
    b = bgr_prime[:, :, 0]
    g = bgr_prime[:, :, 1]
    r = bgr_prime[:, :, 2]
    
    # Calculate K (Black)
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    
    # Avoid division by zero
    k_safe = np.where(k == 1.0, 1.0 - 1e-6, k)
    
    # Calculate C, M, Y
    c = (1.0 - r - k_safe) / (1.0 - k_safe)
    m = (1.0 - g - k_safe) / (1.0 - k_safe)
    y = (1.0 - b - k_safe) / (1.0 - k_safe)
    
    # Clean up division by zero artifacts
    c[k == 1.0] = 0
    m[k == 1.0] = 0
    y[k == 1.0] = 0
    
    # Apply a gamma/contrast curve so light areas drop to zero faster (removes noise)
    # and dark areas get darker.
    gamma = 1.5
    c = np.power(c, gamma)
    m = np.power(m, gamma)
    y = np.power(y, gamma)
    k = np.power(k, gamma)
    
    return {"C": c, "M": m, "Y": y, "K": k}


def generate_wave_paths(intensity_map, angle_deg, spacing, max_amplitude, frequency, resolution):
    """
    Projects parallel lines across an intensity map, modulating into sine waves
    based on the pixel darkness underneath.
    
    intensity_map: 2D numpy array (w, h) where 0 = white, 1 = dark
    angle_deg: angle of the parallel lines
    """
    h, w = intensity_map.shape
    paths = []
    
    # Convert angle to radians
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    
    # Diagonal length to ensure lines cover everything after rotation
    diag = math.hypot(w, h)
    cx, cy = w / 2.0, h / 2.0
    
    # Generate parallel line offsets
    y_offsets = np.arange(-diag / 2.0, diag / 2.0, spacing)
    
    for y_off in y_offsets:
        current_path = []
        is_drawing = False
        
        # Step along the line
        for x_off in np.arange(-diag / 2.0, diag / 2.0, resolution):
            
            # 1. Base point on the un-modulated parallel line (rotated backwards to image space)
            px = cx + x_off * cos_theta - y_off * sin_theta
            py = cy + x_off * sin_theta + y_off * cos_theta
            
            # Check bounds
            if 0 <= px < w and 0 <= py < h:
                
                # 2. Sample Image Intensity
                intensity = intensity_map[int(py), int(px)]
                
                # 3. Cutoff threshold (don't draw in empty space to save plotter time)
                # INCREASED to 0.15 to aggressively trim background noise and floating dots
                if intensity < 0.15:
                    if is_drawing:
                        # Close out the current path
                        if len(current_path) > 1:
                            paths.append(current_path)
                        current_path = []
                        is_drawing = False
                    continue
                else:
                    is_drawing = True
                
                # 4. Amplitude Modulation (The Magic)
                # Oscillation magnitude depends on darkness.
                amp = max_amplitude * intensity
                
                # The wave oscillates perpendicular to the direction of travel
                wave_offset = amp * math.sin(x_off * frequency)
                
                # Add perpendicular offset
                wx = px - wave_offset * sin_theta
                wy = py + wave_offset * cos_theta
                
                current_path.append((wx, wy))
                
        # If line ended but we were drawing, save it
        if len(current_path) > 1:
            paths.append(current_path)
            
    return paths

# --------------------------
# Utility: SVG Export
# --------------------------
def export_cmyk_svg(layers_paths, img_size, out_path):
    iw, ih = img_size
    doc = Document()
    svg = doc.createElement('svg')
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    svg.setAttribute('width', str(iw))
    svg.setAttribute('height', str(ih))
    svg.setAttribute('viewBox', f"0 0 {iw} {ih}")
    doc.appendChild(svg)

    # CMYK colors for rendering
    colors = {
        "C": "#00FFFF",
        "M": "#FF00FF",
        "Y": "#FFFF00",
        "K": "#000000"
    }

    for channel, paths in layers_paths.items():
        if not paths: continue
        
        g = doc.createElement('g')
        g.setAttribute('id', f"Pen_{channel}")
        g.setAttribute('stroke', colors[channel])
        g.setAttribute('stroke-width', '1') 
        g.setAttribute('fill', 'none')      
        svg.appendChild(g)

        for path in paths:
            if len(path) < 2: continue
            
            polyline = doc.createElement('polyline')
            points_attr = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in path)
            polyline.setAttribute('points', points_attr)
            
            g.appendChild(polyline)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc.toprettyxml())


class CMYKWavePlotterApp:
    def __init__(self, root):
        self.root = root
        root.title("Plotter Prototype (CMYK Wavy Halftones)")

        self.image_path = None
        self.img_w = 1
        self.img_h = 1
        
        # Will store the output paths separated by CMYK layer
        self.layers_paths = {"C": [], "M": [], "Y": [], "K": []}

        # UI layout
        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        btn_load = ttk.Button(ctrl_frame, text="Load Image (PNG/JPG)", command=self.load_image)
        btn_load.pack(fill=tk.X, pady=4)

        self.lbl_file = ttk.Label(ctrl_frame, text="No file loaded", wraplength=220)
        self.lbl_file.pack(fill=tk.X, pady=2)

        # CMYK Wave parameters
        ttk.Label(ctrl_frame, text="Wave Parameters").pack(anchor=tk.W, pady=(8,0))
        
        self.spacing_var = tk.DoubleVar(value=4.0)
        ttk.Label(ctrl_frame, text="Line Spacing (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.spacing_var).pack(fill=tk.X)

        self.amplitude_var = tk.DoubleVar(value=3.5)
        ttk.Label(ctrl_frame, text="Max Wave Amplitude (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.amplitude_var).pack(fill=tk.X)
        
        self.frequency_var = tk.DoubleVar(value=0.5)
        ttk.Label(ctrl_frame, text="Wave Frequency").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.frequency_var).pack(fill=tk.X)
        
        self.resolution_var = tk.DoubleVar(value=1.0)
        ttk.Label(ctrl_frame, text="Sample Resolution (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.resolution_var).pack(fill=tk.X)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)

        btn_vectorize = ttk.Button(ctrl_frame, text="Generate Waves", command=self.generate_waves)
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
        # using a darker off-white to make the CMYK colors pop
        self.canvas = tk.Canvas(canvas_frame, bg='#f0f0f0', width=700, height=700)
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

    def generate_waves(self):
        if self.image_path is None:
            messagebox.showwarning("No file", "Please load an image first.")
            return
            
        self.canvas.delete("all")
        self.set_status("Separating CMYK and generating continuous waves...")
        self.root.update()

        try:
            # Load and composite alpha to white if needed
            img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                bg = np.ones_like(rgb) * 255.0
                rgb = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
                img = rgb.astype(np.uint8)
                
            img_bgr = img[:, :, :3]
            
            # Separate Channels
            channels = rgb_to_cmyk_channels(img_bgr)
            
            # Fetch Math Params
            spacing = float(self.spacing_var.get())
            amp = float(self.amplitude_var.get())
            freq = float(self.frequency_var.get())
            res = float(self.resolution_var.get())
            
            # Standard drafting angles for CMYK to avoid Moire Patterns
            angles = {"C": 15.0, "M": 75.0, "Y": 0.0, "K": 45.0}
            
            self.layers_paths = {"C": [], "M": [], "Y": [], "K": []}
            
            for channel in ["C", "M", "Y", "K"]:
                self.set_status(f"Generating Waves for {channel} channel...")
                self.root.update()
                
                paths = generate_wave_paths(
                    channels[channel], 
                    angle_deg=angles[channel], 
                    spacing=spacing, 
                    max_amplitude=amp, 
                    frequency=freq, 
                    resolution=res
                )
                self.layers_paths[channel] = paths
                
            self.set_status("Generation complete. Ready to animate or export.")
            self.show_preview_fast()
            
        except Exception as e:
            messagebox.showerror("Algorithm Error", str(e))
            self.set_status("Error calculating waves.")
            
    def show_preview_fast(self):
        """ Quick, non-animated dump to canvas just to see the shape """
        self.canvas.delete("all")
        
        cw = int(self.canvas.winfo_width() or 700)
        ch = int(self.canvas.winfo_height() or 700)
        iw = self.img_w
        ih = self.img_h
        
        # Calculate scaling to fit canvas tightly
        scale_x = (cw - 20) / iw
        scale_y = (ch - 20) / ih
        scale = min(scale_x, scale_y)
        offset_x = (cw - iw*scale) / 2.0
        offset_y = (ch - ih*scale) / 2.0
        
        colors = {"C": "#00FFFF", "M": "#FF00FF", "Y": "#CCCC00", "K": "#000000"} # slightly darker yellow for visibility
        
        for chn, paths in self.layers_paths.items():
            for path in paths:
                coords = []
                for x, y in path:
                    coords.extend((x*scale + offset_x, y*scale + offset_y))
                if len(coords) >= 4:
                    self.canvas.create_line(coords, fill=colors[chn], width=1.0)


    def export_svg(self):
        if not any(self.layers_paths.values()):
            messagebox.showwarning("No vectors", "No paths generated yet. Click 'Generate Waves' first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files","*.svg")])
        if not out:
            return
        try:
            export_cmyk_svg(self.layers_paths, (self.img_w, self.img_h), out)
            messagebox.showinfo("Exported", f"Master SVG exported to: {out}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def start_drawing(self):
        if not any(self.layers_paths.values()):
            messagebox.showwarning("No vectors", "No paths generated yet. Click 'Generate Waves' first.")
            return
            
        self.canvas.delete("all")
        self.set_status("Animating Plotter Progress...")
        
        # Build drawable stack
        draw_list = []
        colors = {"C": "#00FFFF", "M": "#FF00FF", "Y": "#CCCC00", "K": "#000000"} 
        
        cw = int(self.canvas.winfo_width() or 700)
        ch = int(self.canvas.winfo_height() or 700)
        scale_x = (cw - 20) / self.img_w
        scale_y = (ch - 20) / self.img_h
        scale = min(scale_x, scale_y)
        offset_x = (cw - self.img_w*scale) / 2.0
        offset_y = (ch - self.img_h*scale) / 2.0

        for chn, paths in self.layers_paths.items():
            for path in paths: # path is a list of (x,y)
                if len(path) < 2: continue
                
                # Convert to canvas coords
                mapped_path = [(x*scale + offset_x, y*scale + offset_y) for x,y in path]
                
                # Break into line segments
                segs = []
                for i in range(len(mapped_path)-1):
                    segs.append((mapped_path[i], mapped_path[i+1]))
                draw_list.append({"color": colors[chn], "segs": segs})

        self._draw_state = {"draw_list": draw_list, "region_i": 0, "seg_i": 0, "pos": None}
        self.draw_speed = 8.0  # pixels per frame. High because waves are long.
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
            # move to next continuous line
            state["region_i"] += 1
            state["seg_i"] = 0
            state["pos"] = None
            self._anim_job = self.root.after(1, self._animate_step) # Minimal pen-up delay
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

        self.canvas.create_line(ax, ay, nx, ny, fill=region["color"], width=1.5)
        state["pos"] = (nx, ny)

        # if reached end of segment
        if math.hypot(bx - nx, by - ny) < 1e-2:
            state["seg_i"] += 1
            state["pos"] = None

        self._anim_job = self.root.after(1, self._animate_step)

    def clear_canvas(self):
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self.canvas.delete("all")
        self.layers_paths = {"C": [], "M": [], "Y": [], "K": []}
        self.set_status("Cleared")


# --------------------------
# Run the app
# --------------------------
def main():
    root = tk.Tk()
    app = CMYKWavePlotterApp(root)
    root.geometry("1100x760")
    root.mainloop()

if __name__ == "__main__":
    main()
