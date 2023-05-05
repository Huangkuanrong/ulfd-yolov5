import os
import subprocess
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import shutil

# Get the directory path of the current Python file
FILE_PATH = os.path.abspath(__file__) # C:/Users/.../Ultra-Fast-Lane-Detection/main.py
DRIVE_NAME, _ = os.path.splitdrive(FILE_PATH) # C:
BASE_DIR = os.path.dirname(FILE_PATH) # C:/Users/.../Ultra-Fast-Lane-Detection

UFLD_RESULT_DIR = os.path.join(BASE_DIR, 'content', 'result', 'ufld', 'result.txt')
YOLO_RESULT_DIR = os.path.join(BASE_DIR, 'content', 'result', 'yolo', 'result.txt')
JOIN_RESULT_DIR = os.path.join(BASE_DIR, 'content', 'result', 'join-result', 'result.txt')
IMG_DIR = os.path.join(BASE_DIR, 'content', 'img', 'road.jpg')

def choose_image_file():
    global IMG_DIR
    # Open a file dialog to allow the user to select an image file
    file_path = filedialog.askopenfilename()
    # Copy the selected image file to BASE_DIR/content/img/road.jpg
    dest_dir = os.path.join(BASE_DIR, 'content', 'img', 'road.jpg')
    shutil.copy(file_path, dest_dir)
    IMG_DIR = dest_dir
    
# Function to run UFLD command
def run_ufld():
    global IMG_DIR
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/content/Ultra-Fast-Lane-Detection/demo.py content/Ultra-Fast-Lane-Detection/configs/tusimple.py --img {IMG_DIR}'

    try:
        subprocess.run(command, shell=True, check=True)
        with open(UFLD_RESULT_DIR, 'r') as f:
            file_dir = BASE_DIR + "\\" + f.readline().strip() + '.jpg'
            img = cv2.imread(file_dir)

            # Show the image in a new window
            cv2.namedWindow("Ultra Fast Lane Detection Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Ultra Fast Lane Detection Result", img)
            cv2.waitKey(0)

        messagebox.showinfo("Success", "Ultra Fast Lane Detection completed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Ultra Fast Lane Detection failed!")

# Function to run YOLO command
def run_yolo():
    global IMG_DIR
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/content/yolov5/yolov5s.py --img {IMG_DIR}'
    try:
        subprocess.run(command, shell=True, check=True)
        with open(YOLO_RESULT_DIR, 'r') as f:
            file_dir = BASE_DIR + "\\" + f.readline().strip() + '.jpg'
            img = cv2.imread(file_dir)

            # Show the image in a new window
            cv2.namedWindow("YOLOv5 Result", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLOv5 Result", img)
            cv2.waitKey(0)

        messagebox.showinfo("Success", "YOLOv5 completed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "YOLOv5 failed!")

# Create the Join button
def run_join():
    global IMG_DIR
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/join.py --img {IMG_DIR}'
    try:
        subprocess.run(command, shell=True, check=True)
        with open(JOIN_RESULT_DIR, 'r') as f:
            file_dir = BASE_DIR + "\\" + f.readline().strip() + '.jpg'
            img = cv2.imread(file_dir)

            # Show the image in a new window
            cv2.namedWindow("Join Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Join Result", img)
            cv2.waitKey(0)
            
        messagebox.showinfo("Success", "Join completed successfully!")
    except subprocess.CalledProcessError:
        messagebox.showerror("Error", "Join failed!")

# Create the main window
root = tk.Tk()
root.title("Lane Detection Tools")
root.geometry("500x400")

# Create a frame to hold the buttons and image label
frame = tk.Frame(root)
frame.pack()

# Create a frame to hold the Choose Image button and image label
frame_choose_image = tk.Frame(frame)
frame_choose_image.pack(side=tk.TOP, padx=10, pady=10)

# Create the Choose Image button
button_choose_image = tk.Button(frame_choose_image, text="Choose Image", width=25, height=2, command=choose_image_file)
button_choose_image.pack(side=tk.LEFT)

# Create the image label
img_label = tk.Label(frame_choose_image)
img_label.pack(side=tk.LEFT, padx=10)

# Create a frame to hold the UFLD and YOLO buttons and result labels
frame_buttons = tk.Frame(frame)
frame_buttons.pack(side=tk.TOP, padx=10, pady=10)

# Create the UFLD button
button_ufld = tk.Button(frame_buttons, text="Ultra Fast Lane Detection", width=25, height=2, command=run_ufld)
button_ufld.pack(side=tk.LEFT, padx=10)

# Create the YOLO button
button_yolo = tk.Button(frame_buttons, text="YOLOv5", width=25, height=2, command=run_yolo)
button_yolo.pack(side=tk.LEFT, padx=10)

# Create the Join button
button_join = tk.Button(frame_buttons, text="Join", width=25, height=2, command=run_join)
button_join.pack(side=tk.LEFT, padx=10)

root.mainloop()