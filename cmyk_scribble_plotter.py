"""
cmyk_scribble_plotter.py

A plotter-friendly vectorization engine that uses a density-driven random walk 
to simulate an artist scribbling with CMYK pens. The algorithm drops a virtual pen, 
finds the darkest local pixel, draws a line to it, and subtracts that darkness 
from the virtual canvas so it is forced to explore and fill new areas.

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
import random
from xml.dom.minidom import Document

# --------------------------
# Algorithmic Magic: CMYK Separation
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
    
    # Apply a gamma curve to cut out compression/background noise
    gamma = 1.5
    c = np.power(c, gamma)
    m = np.power(m, gamma)
    y = np.power(y, gamma)
    k = np.power(k, gamma)
    
    return {"C": c, "M": m, "Y": y, "K": k}

# --------------------------
# Algorithmic Magic: The Traveling Scribbler 
# --------------------------

def generate_scribble_path(intensity_map, max_nodes=20000, search_radius=10, ink_depletion=0.1):
    """
    Generates a single continuous line that seeks out dark pixels and "erases" them.
    """
    h, w = intensity_map.shape
    path = []
    
    # Make a working copy of the map we can destroy as we draw
    work_map = intensity_map.copy()
    
    # Start the pen in the middle of the paper (or find darkest spot)
    px, py = w // 2, h // 2
    
    # Optional: Find absolute darkest starting pixel globally to drop the pen
    max_idx = np.argmax(work_map)
    py, px = divmod(max_idx, w)
    
    path.append((px, py))
    
    for _ in range(max_nodes):
        # 1. Define local search window 
        min_x = max(0, int(px - search_radius))
        max_x = min(w, int(px + search_radius + 1))
        min_y = max(0, int(py - search_radius))
        max_y = min(h, int(py + search_radius + 1))
        
        # Extract local neighborhood
        window = work_map[min_y:max_y, min_x:max_x]
        
        if window.size == 0 or np.max(window) < 0.05:
            # If the immediate area is perfectly clean, we need to lift the pen and jump
            # to a completely new dirty area to save time dragging lines across white space.
            dark_spots = np.where(work_map > 0.1)
            if len(dark_spots[0]) > 0:
                # Pick a random dark spot to jump to
                idx = random.randint(0, len(dark_spots[0])-1)
                py = dark_spots[0][idx]
                px = dark_spots[1][idx]
                
                # We add a "None" marker to indicate a physical Pen-Up command to the plotter
                path.append(None)
                path.append((px, py))
                continue
            else:
                # The entire paper is clean. We are done!
                break
                
        # 2. Add an optical distance penalty. 
        # We prefer walking to a dark pixel right next to us over a dark pixel 9 pixels away.
        # This prevents the scribble from looking like a chaotic starburst.
        y_coords, x_coords = np.indices(window.shape)
        
        # Absolute coordinates of window pixels
        abs_y = y_coords + min_y
        abs_x = x_coords + min_x
        
        # Distance squared to current pen position
        dist_sq = (abs_x - px)**2 + (abs_y - py)**2
        
        # Avoid dividing by zero at the pen's current position
        dist_sq[dist_sq == 0] = 1.0 
        
        # Score = Darkness / Distance. 
        # (We add tiny random noise so the pen doesn't get trapped in a deterministic 2-pixel oscillation)
        noise = np.random.uniform(0.9, 1.1, window.shape)
        scores = (window * noise) / dist_sq
        
        # 3. Find the winner
        winner_idx = np.argmax(scores)
        wy, wx = divmod(winner_idx, window.shape[1])
        
        # Update pen position
        px = wx + min_x
        py = wy + min_y
        
        path.append((px, py))
        
        # 4. Erase the ink from the paper!
        # The plotter just "shaded" this pixel, so the image needs it less now.
        # We deplete the darkness, forcing the pen to move on next frame.
        work_map[py, px] = max(0.0, work_map[py, px] - ink_depletion)

    return path

# --------------------------
# Utility: SVG Export
# --------------------------
def export_cmyk_scribble_svg(layers_paths, img_size, out_path):
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

        # The scribble generator can output lists with `None` to indicate Pen-Up.
        # We need to split those into discrete <polyline> blocks.
        
        current_segment = []
        
        for pt in paths:
            if pt is None:
                if len(current_segment) >= 2:
                    polyline = doc.createElement('polyline')
                    points_attr = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in current_segment)
                    polyline.setAttribute('points', points_attr)
                    g.appendChild(polyline)
                current_segment = []
            else:
                current_segment.append(pt)
                
        # Catch the final segment
        if len(current_segment) >= 2:
            polyline = doc.createElement('polyline')
            points_attr = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in current_segment)
            polyline.setAttribute('points', points_attr)
            g.appendChild(polyline)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc.toprettyxml())


# --------------------------
# GUI Application
# --------------------------
class CMYKScribbleApp:
    def __init__(self, root):
        self.root = root
        root.title("Plotter Prototype (Organic CMYK Scribble)")

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

        # Scribble parameters
        ttk.Label(ctrl_frame, text="Virtual Artist Parameters").pack(anchor=tk.W, pady=(8,0))
        
        self.max_nodes_var = tk.IntVar(value=30000)
        ttk.Label(ctrl_frame, text="Max Pen Strokes per Color").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.max_nodes_var).pack(fill=tk.X)

        self.radius_var = tk.IntVar(value=6)
        ttk.Label(ctrl_frame, text="Pen Jump Radius (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.radius_var).pack(fill=tk.X)
        
        self.depletion_var = tk.DoubleVar(value=0.25)
        ttk.Label(ctrl_frame, text="Ink Depletion Rate (0-1.0)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.depletion_var).pack(fill=tk.X)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)

        btn_vectorize = ttk.Button(ctrl_frame, text="Generate Scribble", command=self.generate_scribble)
        btn_vectorize.pack(fill=tk.X, pady=4)

        btn_export = ttk.Button(ctrl_frame, text="Export Vector to SVG", command=self.export_svg)
        btn_export.pack(fill=tk.X, pady=4)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)

        btn_draw = ttk.Button(ctrl_frame, text="Animate Drawing", command=self.start_drawing)
        btn_draw.pack(fill=tk.X, pady=4)

        btn_clear = ttk.Button(ctrl_frame, text="Clear Canvas", command=self.clear_canvas)
        btn_clear.pack(fill=tk.X, pady=4)

        # Canvas
        canvas_frame = ttk.Frame(root)
        canvas_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        self.canvas = tk.Canvas(canvas_frame, bg='#f0f0f0', width=700, height=700)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # status bar
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

    def generate_scribble(self):
        if self.image_path is None:
            messagebox.showwarning("No file", "Please load an image first.")
            return
            
        self.canvas.delete("all")
        self.set_status("Separating CMYK and generating chaotic scribbles...")
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
            
            channels = rgb_to_cmyk_channels(img_bgr)
            
            nodes = int(self.max_nodes_var.get())
            radius = int(self.radius_var.get())
            depletion = float(self.depletion_var.get())
            
            self.layers_paths = {"C": [], "M": [], "Y": [], "K": []}
            
            for channel in ["C", "M", "Y", "K"]:
                self.set_status(f"Running Virtual Pen for {channel} channel...")
                self.root.update()
                
                path = generate_scribble_path(
                    channels[channel], 
                    max_nodes=nodes,
                    search_radius=radius,
                    ink_depletion=depletion
                )
                self.layers_paths[channel] = path
                
            self.set_status("Generation complete. Ready to animate or export.")
            self.show_preview_fast()
            
        except Exception as e:
            messagebox.showerror("Algorithm Error", str(e))
            self.set_status("Error calculating scribbles.")
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
        
        colors = {"C": "#00FFFF", "M": "#FF00FF", "Y": "#CCCC00", "K": "#000000"} 
        
        for chn, path in self.layers_paths.items():
            if not path: continue
            
            current_seg = []
            for pt in path:
                if pt is None:
                    if len(current_seg) >= 4:
                        self.canvas.create_line(current_seg, fill=colors[chn], width=1.0)
                    current_seg = []
                else:
                    x, y = pt
                    current_seg.extend((x*scale + offset_x, y*scale + offset_y))
                    
            if len(current_seg) >= 4:
                self.canvas.create_line(current_seg, fill=colors[chn], width=1.0)

    def export_svg(self):
        if not any(self.layers_paths.values()):
            messagebox.showwarning("No vectors", "No paths generated yet. Click 'Generate Scribble' first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files","*.svg")])
        if not out:
            return
        try:
            export_cmyk_scribble_svg(self.layers_paths, (self.img_w, self.img_h), out)
            messagebox.showinfo("Exported", f"Master SVG exported to: {out}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def start_drawing(self):
        if not any(self.layers_paths.values()):
            messagebox.showwarning("No vectors", "No paths generated yet. Click 'Generate' first.")
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

        for chn, path in self.layers_paths.items():
            if not path: continue
            
            current_segs = []
            # We must break the single long array into discrete point-to-point tuples for the animator
            last_pt = None
            for pt in path: 
                if pt is None:
                    last_pt = None
                    continue
                    
                mapped = (pt[0]*scale + offset_x, pt[1]*scale + offset_y)
                if last_pt is not None:
                    current_segs.append((last_pt, mapped))
                last_pt = mapped
                
            if current_segs:
                draw_list.append({"color": colors[chn], "segs": current_segs})

        self._draw_state = {"draw_list": draw_list, "region_i": 0, "seg_i": 0, "pos": None}
        self.draw_speed = 30.0  # Pixels per frame. Extremely fast because the lines are tiny
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
            state["region_i"] += 1
            state["seg_i"] = 0
            state["pos"] = None
            self._anim_job = self.root.after(1, self._animate_step) 
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
    app = CMYKScribbleApp(root)
    root.geometry("1100x760")
    root.mainloop()

if __name__ == "__main__":
    main()
