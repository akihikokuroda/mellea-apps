#!/usr/bin/env python3
"""
Simple Local Photo Viewer - Display photos using tkinter (no browser needed)
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import os
import sys

class PhotoViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Viewer")
        self.root.geometry("900x700")
        self.root.minsize(600, 400)

        self.photos = []
        self.current_index = 0
        self.photo_image = None
        self.is_fullscreen = False

        self.setup_ui()
        self.root.after(500, self.load_default_directory)

    def setup_ui(self):
        control_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        control_frame.pack(fill=tk.X)

        tk.Label(control_frame, text="Photo Directory:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)

        self.path_entry = tk.Entry(control_frame, width=50)
        self.path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        tk.Button(control_frame, text="Browse", command=self.browse_directory).pack(side=tk.LEFT, padx=2)
        tk.Button(control_frame, text="Load", command=self.load_photos).pack(side=tk.LEFT, padx=2)

        self.image_label = tk.Label(self.root, bg="black", text="Loading...", fg="white")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        nav_frame = tk.Frame(self.root, bg="#f0f0f0", padx=10, pady=10)
        nav_frame.pack(fill=tk.X)

        tk.Button(nav_frame, text="Previous", command=self.show_previous).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next", command=self.show_next).pack(side=tk.LEFT, padx=5)

        self.info_label = tk.Label(nav_frame, text="No photos loaded", bg="#f0f0f0")
        self.info_label.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)

        tk.Button(nav_frame, text="Fullscreen", command=self.toggle_fullscreen).pack(side=tk.RIGHT, padx=5)
        tk.Button(nav_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)

        self.root.bind('<Right>', lambda e: self.show_next())
        self.root.bind('<Left>', lambda e: self.show_previous())
        self.root.bind('<f>', lambda e: self.toggle_fullscreen())
        self.root.bind('<F>', lambda e: self.toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.exit_fullscreen())
        self.root.bind('<q>', lambda e: self.root.quit())
        self.root.bind('<Q>', lambda e: self.root.quit())

    def load_default_directory(self):
        pictures_dir = os.path.expanduser("~/Pictures")
        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, pictures_dir)
        self.load_photos()

    def browse_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, directory)
            self.load_photos()

    def load_photos(self):
        directory = self.path_entry.get()

        try:
            path = Path(directory).expanduser().resolve()
            if not path.exists() or not path.is_dir():
                messagebox.showerror("Error", "Directory not found")
                return

            extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff'}
            self.photos = sorted([
                str(f) for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ], key=lambda x: Path(x).name.lower())

            if not self.photos:
                messagebox.showinfo("Info", "No photos found in this directory")
                self.current_index = 0
                self.display_empty()
                return

            self.current_index = 0
            self.display_photo()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load photos: {e}")

    def display_photo(self):
        if not self.photos:
            self.display_empty()
            return

        try:
            photo_path = self.photos[self.current_index]
            image = Image.open(photo_path)

            window_width = self.image_label.winfo_width()
            window_height = self.image_label.winfo_height()

            if window_width <= 1:
                window_width = 800
                window_height = 600

            image.thumbnail((window_width, window_height), Image.Resampling.LANCZOS)

            self.photo_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo_image, text="")

            filename = Path(photo_path).name
            self.info_label.config(
                text=f"[{self.current_index + 1}/{len(self.photos)}] {filename}"
            )
        except Exception as e:
            self.image_label.config(text=f"Error loading image: {e}", bg="black")

    def display_empty(self):
        self.image_label.config(text="No photos loaded", image="")
        self.info_label.config(text="Load a directory to view photos")

    def show_next(self):
        if self.photos:
            self.current_index = (self.current_index + 1) % len(self.photos)
            self.display_photo()

    def show_previous(self):
        if self.photos:
            self.current_index = (self.current_index - 1) % len(self.photos)
            self.display_photo()

    def toggle_fullscreen(self):
        self.is_fullscreen = not self.is_fullscreen
        self.root.attributes('-fullscreen', self.is_fullscreen)

    def exit_fullscreen(self):
        if self.is_fullscreen:
            self.is_fullscreen = False
            self.root.attributes('-fullscreen', False)


if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = PhotoViewer(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
