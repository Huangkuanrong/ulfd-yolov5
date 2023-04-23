import os
import subprocess
import tkinter

# Get the directory path of the current Python file
FILE_PATH = os.path.abspath(__file__) # C:/Users/.../Ultra-Fast-Lane-Detection/main.py
DRIVE_NAME, _ = os.path.splitdrive(FILE_PATH) # C:
BASE_DIR = os.path.dirname(FILE_PATH) # C:/Users/.../Ultra-Fast-Lane-Detection

def run_ufld():
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/content/Ultra-Fast-Lane-Detection/demo.py {BASE_DIR}/content/Ultra-Fast-Lane-Detection/configs/tusimple.py --test_model {BASE_DIR}/content/Ultra-Fast-Lane-Detection/tusimple_18.pth'
    subprocess.run(command, shell=True)

def run_yolo():
    # Activate the conda environment and run the command
    command = f'cd {BASE_DIR} && {DRIVE_NAME} && conda activate lane-det-yolo && python {BASE_DIR}/content/yolov5/yolov5s.py'
    subprocess.run(command, shell=True)

root = tkinter.Tk()

# Call the run_command() function when the button is clicked
button_ufld = tkinter.Button(root, text="Ultra Fast Lane Detection", command=run_ufld)
button_ufld.pack()

# Call the run_command() function when the button is clicked
button_yolo = tkinter.Button(root, text="Yolov5", command=run_yolo)
button_yolo.pack()

root.mainloop()