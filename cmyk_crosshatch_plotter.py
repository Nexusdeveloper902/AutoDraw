import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import math
from xml.dom.minidom import Document

def rgb_to_cmyk_channels(img_bgr, gamma=1.5):
    bgr_prime = img_bgr.astype(np.float32) / 255.0
    b = bgr_prime[:, :, 0]
    g = bgr_prime[:, :, 1]
    r = bgr_prime[:, :, 2]
    
    k = 1.0 - np.maximum(np.maximum(r, g), b)
    k_safe = np.where(k == 1.0, 1.0 - 1e-6, k)
    
    c = (1.0 - r - k_safe) / (1.0 - k_safe)
    m = (1.0 - g - k_safe) / (1.0 - k_safe)
    y = (1.0 - b - k_safe) / (1.0 - k_safe)
    
    c[k == 1.0] = 0
    m[k == 1.0] = 0
    y[k == 1.0] = 0
    
    c = np.power(c, gamma)
    m = np.power(m, gamma)
    y = np.power(y, gamma)
    k = np.power(k, gamma)
    
    return {"C": c, "M": m, "Y": y, "K": k}

def generate_hatch_lines(intensity_map, angle_deg, spacing, threshold, resolution=1.0):
    h, w = intensity_map.shape
    paths = []
    theta = math.radians(angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    diag = math.hypot(w, h)
    cx, cy = w / 2.0, h / 2.0
    
    y_offsets = np.arange(-diag / 2.0, diag / 2.0, spacing)
    
    for y_off in y_offsets:
        start_pt = None
        last_pt = None
        
        for x_off in np.arange(-diag / 2.0, diag / 2.0, resolution):
            px = cx + x_off * cos_theta - y_off * sin_theta
            py = cy + x_off * sin_theta + y_off * cos_theta
            
            if 0 <= px < w and 0 <= py < h:
                intensity = intensity_map[int(py), int(px)]
                if intensity >= threshold:
                    if start_pt is None:
                        start_pt = (px, py)
                    last_pt = (px, py)
                else:
                    if start_pt is not None and last_pt is not None:
                        if math.hypot(last_pt[0]-start_pt[0], last_pt[1]-start_pt[1]) > 1.0:
                            paths.append([start_pt, last_pt])
                    start_pt = None
                    last_pt = None
            else:
                if start_pt is not None and last_pt is not None:
                    if math.hypot(last_pt[0]-start_pt[0], last_pt[1]-start_pt[1]) > 1.0:
                        paths.append([start_pt, last_pt])
                start_pt = None
                last_pt = None
                
        if start_pt is not None and last_pt is not None:
            if math.hypot(last_pt[0]-start_pt[0], last_pt[1]-start_pt[1]) > 1.0:
                paths.append([start_pt, last_pt])
            
    return paths

def generate_layered_crosshatch(intensity_map, base_angle, spacing, thresholds):
    paths = []
    paths.extend(generate_hatch_lines(intensity_map, base_angle, spacing, thresholds[0]))
    paths.extend(generate_hatch_lines(intensity_map, base_angle + 90, spacing, thresholds[1]))
    paths.extend(generate_hatch_lines(intensity_map, base_angle + 45, spacing, thresholds[2]))
    paths.extend(generate_hatch_lines(intensity_map, base_angle + 135, spacing, thresholds[3]))
    return paths

def export_crosshatch_svg(layers_paths, img_size, out_path):
    iw, ih = img_size
    doc = Document()
    svg = doc.createElement('svg')
    svg.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    svg.setAttribute('width', str(iw))
    svg.setAttribute('height', str(ih))
    svg.setAttribute('viewBox', f"0 0 {iw} {ih}")
    doc.appendChild(svg)

    colors = {"C": "#00FFFF", "M": "#FF00FF", "Y": "#FFFF00", "K": "#000000"}

    for channel, paths in layers_paths.items():
        if not paths: continue
        g = doc.createElement('g')
        g.setAttribute('id', f"Pen_{channel}")
        g.setAttribute('stroke', colors[channel])
        g.setAttribute('stroke-width', '1') 
        g.setAttribute('fill', 'none')      
        svg.appendChild(g)

        for path in paths:
            if len(path) == 2:
                line = doc.createElement('line')
                line.setAttribute('x1', f"{path[0][0]:.2f}")
                line.setAttribute('y1', f"{path[0][1]:.2f}")
                line.setAttribute('x2', f"{path[1][0]:.2f}")
                line.setAttribute('y2', f"{path[1][1]:.2f}")
                g.appendChild(line)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(doc.toprettyxml())


class CMYKCrosshatchApp:
    def __init__(self, root):
        self.root = root
        root.title("Plotter Prototype (CMYK Density Cross-Hatch)")

        self.image_path = None
        self.img_w = 1
        self.img_h = 1
        
        self.layers_paths = {"C": [], "M": [], "Y": [], "K": []}

        ctrl_frame = ttk.Frame(root)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)

        btn_load = ttk.Button(ctrl_frame, text="Load Image (PNG/JPG)", command=self.load_image)
        btn_load.pack(fill=tk.X, pady=4)

        self.lbl_file = ttk.Label(ctrl_frame, text="No file loaded", wraplength=220)
        self.lbl_file.pack(fill=tk.X, pady=2)

        ttk.Label(ctrl_frame, text="Cross-Hatch Parameters").pack(anchor=tk.W, pady=(8,0))
        
        self.spacing_var = tk.DoubleVar(value=4.0)
        ttk.Label(ctrl_frame, text="Line Spacing (px)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.spacing_var).pack(fill=tk.X)

        self.t1_var = tk.DoubleVar(value=0.15)
        ttk.Label(ctrl_frame, text="Layer 1 Threshold (>X)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.t1_var).pack(fill=tk.X)
        
        self.t2_var = tk.DoubleVar(value=0.40)
        ttk.Label(ctrl_frame, text="Layer 2 Threshold (>X)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.t2_var).pack(fill=tk.X)

        self.t3_var = tk.DoubleVar(value=0.65)
        ttk.Label(ctrl_frame, text="Layer 3 Threshold (>X)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.t3_var).pack(fill=tk.X)
        
        self.t4_var = tk.DoubleVar(value=0.85)
        ttk.Label(ctrl_frame, text="Layer 4 Threshold (>X)").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.t4_var).pack(fill=tk.X)

        self.gamma_var = tk.DoubleVar(value=1.5)
        ttk.Label(ctrl_frame, text="Gamma Contrast").pack(anchor=tk.W)
        ttk.Entry(ctrl_frame, textvariable=self.gamma_var).pack(fill=tk.X)

        ttk.Separator(ctrl_frame).pack(fill=tk.X, pady=6)

        btn_vectorize = ttk.Button(ctrl_frame, text="Generate Hatches", command=self.generate_crosshatch)
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
        self.canvas = tk.Canvas(canvas_frame, bg='#f0f0f0', width=700, height=700)
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

    def generate_crosshatch(self):
        if self.image_path is None:
            messagebox.showwarning("No file", "Please load an image first.")
            return
            
        self.canvas.delete("all")
        self.set_status("Separating CMYK and generating engraving hatched lines...")
        self.root.update()

        try:
            img = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:
                alpha = img[:, :, 3] / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                bg = np.ones_like(rgb) * 255.0
                rgb = rgb * alpha[..., None] + bg * (1 - alpha[..., None])
                img = rgb.astype(np.uint8)
                
            img_bgr = img[:, :, :3]
            gamma = float(self.gamma_var.get())
            channels = rgb_to_cmyk_channels(img_bgr, gamma=gamma)
            
            spacing = float(self.spacing_var.get())
            thresholds = [
                float(self.t1_var.get()),
                float(self.t2_var.get()),
                float(self.t3_var.get()),
                float(self.t4_var.get())
            ]
            
            angles = {"C": 15.0, "M": 75.0, "Y": 0.0, "K": 45.0}
            self.layers_paths = {"C": [], "M": [], "Y": [], "K": []}
            
            for channel in ["C", "M", "Y", "K"]:
                self.set_status(f"Hatching {channel} channel...")
                self.root.update()
                
                paths = generate_layered_crosshatch(
                    channels[channel], 
                    base_angle=angles[channel], 
                    spacing=spacing, 
                    thresholds=thresholds
                )
                self.layers_paths[channel] = paths
                
            self.set_status("Generation complete. Ready to animate or export.")
            self.show_preview_fast()
            
        except Exception as e:
            messagebox.showerror("Algorithm Error", str(e))
            self.set_status("Error calculating lines.")
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
        
        for chn, paths in self.layers_paths.items():
            for path in paths:
                if len(path) == 2:
                    p1x = path[0][0]*scale + offset_x
                    p1y = path[0][1]*scale + offset_y
                    p2x = path[1][0]*scale + offset_x
                    p2y = path[1][1]*scale + offset_y
                    self.canvas.create_line(p1x, p1y, p2x, p2y, fill=colors[chn], width=1.0)

    def export_svg(self):
        if not any(self.layers_paths.values()):
            messagebox.showwarning("No vectors", "No paths generated yet. Click 'Generate Hatches' first.")
            return
        out = filedialog.asksaveasfilename(defaultextension=".svg", filetypes=[("SVG files","*.svg")])
        if not out:
            return
        try:
            export_crosshatch_svg(self.layers_paths, (self.img_w, self.img_h), out)
            messagebox.showinfo("Exported", f"Master SVG exported to: {out}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def start_drawing(self):
        if not any(self.layers_paths.values()):
            messagebox.showwarning("No vectors", "No paths generated yet. Click 'Generate' first.")
            return
            
        self.canvas.delete("all")
        self.set_status("Animating Plotter Progress...")
        
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
            for path in paths:
                if len(path) == 2:
                    p1x = path[0][0]*scale + offset_x
                    p1y = path[0][1]*scale + offset_y
                    p2x = path[1][0]*scale + offset_x
                    p2y = path[1][1]*scale + offset_y
                    draw_list.append((colors[chn], p1x, p1y, p2x, p2y))

        self._draw_state = {"list": draw_list, "idx": 0}
        self.draw_speed = 5
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
            color, x1, y1, x2, y2 = dl[i]
            self.canvas.create_line(x1, y1, x2, y2, fill=color, width=1.5)
            
        state["idx"] = end_idx
        self._anim_job = self.root.after(1, self._animate_step)

    def clear_canvas(self):
        if self._anim_job:
            self.root.after_cancel(self._anim_job)
            self._anim_job = None
        self.canvas.delete("all")
        self.layers_paths = {"C": [], "M": [], "Y": [], "K": []}
        self.set_status("Cleared")

def main():
    root = tk.Tk()
    app = CMYKCrosshatchApp(root)
    root.geometry("1100x760")
    root.mainloop()

if __name__ == "__main__":
    main()
