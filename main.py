import os
import subprocess
import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk

# Get the directory path of the current Python file
FILE_PATH = os.path.abspath(__file__) # C:/Users/.../Ultra-Fast-Lane-Detection/main.py
DRIVE_NAME, _ = os.path.splitdrive(FILE_PATH) # C:
BASE_DIR = os.path.dirname(FILE_PATH) # C:/Users/.../Ultra-Fast-Lane-Detection

UFLD_RESULT_DIR = os.path.join(BASE_DIR, 'content', 'result', 'ufld', 'result.txt')
YOLO_RESULT_DIR = os.path.join(BASE_DIR, 'content', 'result', 'yolo', 'result.txt')

# Function to run UFLD command
def run_ufld():
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/content/Ultra-Fast-Lane-Detection/demo.py {BASE_DIR}/content/Ultra-Fast-Lane-Detection/configs/tusimple.py --test_model {BASE_DIR}/content/Ultra-Fast-Lane-Detection/tusimple_18.pth'
    try:
        subprocess.run(command, shell=True, check=True)
        with open(UFLD_RESULT_DIR, 'r') as f:
            file_dir = BASE_DIR + "\\" + f.readline().strip() + '.jpg'
            img = cv2.imread(file_dir)
            
            # Create a new label widget and show the image below the buttons
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            max_size = (300, 300)
            img.thumbnail(max_size, Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            img_label = tk.Label(frame, image=img)
            img_label.image = img  # prevent garbage collection
            img_label.pack(side=tk.LEFT, padx=10, pady=30)
            
        messagebox.showinfo("Success", "Ultra Fast Lane Detection completed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Ultra Fast Lane Detection failed!")

# Function to run YOLO command
def run_yolo():
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/content/yolov5/yolov5s.py'
    try:
        subprocess.run(command, shell=True, check=True)
        with open(YOLO_RESULT_DIR, 'r') as f:
            file_dir = BASE_DIR + "\\" + f.readline().strip() + '.jpg'
            img = cv2.imread(file_dir)
            
            # Create a new label widget and show the image below the buttons
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            max_size = (300, 300)
            img.thumbnail(max_size, Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            img_label = tk.Label(frame, image=img)
            img_label.image = img  # prevent garbage collection
            img_label.pack(side=tk.RIGHT, padx=10, pady=30)
            
        messagebox.showinfo("Success", "YOLOv5 completed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "YOLOv5 failed!")

# Create the main window
root = tk.Tk()
root.title("Lane Detection Tools")
root.geometry("600x400")

# Create a frame to hold the buttons and image label
frame = tk.Frame(root)
frame.pack()

# Create the UFLD button
button_ufld = tk.Button(frame, text="Ultra Fast Lane Detection", width=25, height=2, command=run_ufld)
button_ufld.pack(side=tk.LEFT, padx=10, pady=10)

# Create the YOLO button
button_yolo = tk.Button(frame, text="YOLOv5", width=25, height=2, command=run_yolo)
button_yolo.pack(side=tk.RIGHT, padx=10, pady=10)

# Start the main loop
root.mainloop()