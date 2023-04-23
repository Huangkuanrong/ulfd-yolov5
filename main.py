import os
import subprocess
import tkinter as tk
from tkinter import messagebox

# Get the directory path of the current Python file
FILE_PATH = os.path.abspath(__file__) # C:/Users/.../Ultra-Fast-Lane-Detection/main.py
DRIVE_NAME, _ = os.path.splitdrive(FILE_PATH) # C:
BASE_DIR = os.path.dirname(FILE_PATH) # C:/Users/.../Ultra-Fast-Lane-Detection

# Function to run UFLD command
def run_ufld():
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/content/Ultra-Fast-Lane-Detection/demo.py {BASE_DIR}/content/Ultra-Fast-Lane-Detection/configs/tusimple.py --test_model {BASE_DIR}/content/Ultra-Fast-Lane-Detection/tusimple_18.pth'
    try:
        subprocess.run(command, shell=True, check=True)
        messagebox.showinfo("Success", "Ultra Fast Lane Detection completed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Ultra Fast Lane Detection failed!")

# Function to run YOLO command
def run_yolo():
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/content/yolov5/yolov5s.py'
    try:
        subprocess.run(command, shell=True, check=True)
        messagebox.showinfo("Success", "YOLOv5 completed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "YOLOv5 failed!")

# Create the main window
root = tk.Tk()
root.title("Lane Detection Tools")
root.geometry("400x200")

# Create the UFLD button
button_ufld = tk.Button(root, text="Ultra Fast Lane Detection", width=25, height=2, command=run_ufld)
button_ufld.pack(pady=10)

# Create the YOLO button
button_yolo = tk.Button(root, text="YOLOv5", width=25, height=2, command=run_yolo)
button_yolo.pack(pady=10)

# Start the main loop
root.mainloop()
