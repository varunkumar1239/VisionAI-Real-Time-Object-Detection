
import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
import cv2
import time
import os
import csv
import threading
import queue
from datetime import datetime
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import Counter

# Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ---------- SPLASH SCREEN AND MODEL LOADING ----------

import sys

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "yolov8n.pt")
logo_path = os.path.join(BASE_DIR, "VisionAI_icon.png")

# Create splash window
splash = ctk.CTk()
splash.title("VisionAI")
splash.geometry("500x380")
splash.resizable(False, False)
splash.configure(fg_color="#0B1120")

# Center splash window
splash.update_idletasks()
screen_width = splash.winfo_screenwidth()
screen_height = splash.winfo_screenheight()
x = (screen_width - 500) // 2
y = (screen_height - 380) // 2
splash.geometry(f"500x380+{x}+{y}")

# Application logo
splash_logo = Image.open(logo_path)
splash_logo = splash_logo.resize(
    (110, 110), Image.Resampling.LANCZOS
)
splash_photo = ImageTk.PhotoImage(splash_logo)

ctk.CTkLabel(
    splash,
    image=splash_photo,
    text=""
).pack(pady=(25, 8))

ctk.CTkLabel(
    splash,
    text="VISION AI",
    font=("Arial", 30, "bold"),
    text_color="#38BDF8"
).pack(pady=5)

ctk.CTkLabel(
    splash,
    text="REAL-TIME OBJECT DETECTION SYSTEM",
    font=("Arial", 13, "bold"),
    text_color="white"
).pack(pady=5)

loading_label = ctk.CTkLabel(
    splash,
    text="Initializing AI Model...",
    font=("Arial", 13),
    text_color="#94A3B8"
)
loading_label.pack(pady=(20, 10))

progress_bar = ctk.CTkProgressBar(
    splash,
    width=300,
    mode="indeterminate",
    progress_color="#38BDF8"
)
progress_bar.pack(pady=5)
progress_bar.start()

# Load model in background thread
model_queue = queue.Queue()

def load_model():
    try:
        model_queue.put(("success", YOLO(model_path)))
    except Exception as error:
        model_queue.put(("error", error))

model_thread = threading.Thread(
    target=load_model,
    daemon=True
)
model_thread.start()

# Keep splash screen responsive while model loads
while model_thread.is_alive():
    splash.update()
    time.sleep(0.05)

splash.update()
progress_bar.stop()

result_type, result_value = model_queue.get()

splash.destroy()

if result_type == "error":
    error_root = tk.Tk()
    error_root.withdraw()
    messagebox.showerror(
        "Model Loading Error",
        f"Could not load YOLO model:\n{result_value}"
    )
    error_root.destroy()
    raise result_value

model = result_value

# ---------- END SPLASH SCREEN ----------


# Main window
root = ctk.CTk()
root.title("VisionAI - Object Detection")
icon_path = os.path.join(BASE_DIR, "VisionAI.ico")

if os.path.exists(icon_path):
    root.iconbitmap(icon_path)
root.iconbitmap(os.path.join(BASE_DIR, "VisionAI.ico"))
root.geometry("1250x780")
root.minsize(1050, 700)
root.configure(fg_color="#0B1120")

# Variables
cap = None
out = None
running = False
recording = False
prev_time = time.time()
latest_frame = None

# Detection history variables
detection_history = []
seen_track_ids = set()
last_history_time = {}
logo_image = Image.open(
    os.path.join(BASE_DIR, "VisionAI_icon.png")
)
logo_image = logo_image.resize(
    (55, 55), Image.Resampling.LANCZOS
)
logo_photo = ImageTk.PhotoImage(logo_image)
# ---------- HEADER ----------

header = ctk.CTkFrame(
    root, height=75, fg_color="#111C32", corner_radius=12
)
header.pack(fill="x", padx=20, pady=(15, 10))
header.pack_propagate(False)
# Header logo
header_logo = ctk.CTkLabel(
    header,
    image=logo_photo,
    text=""
)
header_logo.pack(side="left", padx=(20, 5), pady=5)
ctk.CTkLabel(
    header,
    text="VISION AI",
    font=("Arial", 25, "bold"),
    text_color="#38BDF8"
).pack(side="left", padx=25, pady=15)

ctk.CTkLabel(
    header,
    text="REAL-TIME OBJECT DETECTION",
    font=("Arial", 16, "bold"),
    text_color="white"
).pack(side="left", padx=15)

# ---------- MAIN CONTENT ----------

main = ctk.CTkFrame(root, fg_color="transparent")
main.pack(fill="both", expand=True, padx=20, pady=10)

# Camera panel
camera_panel = ctk.CTkFrame(
    main, fg_color="#111C32", corner_radius=15
)
camera_panel.pack(
    side="left", fill="both", expand=True, padx=(0, 12)
)

ctk.CTkLabel(
    camera_panel,
    text="LIVE CAMERA FEED",
    font=("Arial", 18, "bold")
).pack(pady=15)

camera_screen = tk.Label(
    camera_panel,
    text="CAMERA NOT STARTED",
    bg="#080F1D",
    fg="#64748B",
    font=("Arial", 20, "bold")
)
camera_screen.pack(
    fill="both", expand=True, padx=15, pady=(5, 15)
)

# ---------- RIGHT SIDEBAR ----------

sidebar = ctk.CTkFrame(
    main, width=300, fg_color="#111C32", corner_radius=15
)
sidebar.pack(side="right", fill="y")
sidebar.pack_propagate(False)

ctk.CTkLabel(
    sidebar,
    text="ANALYTICS",
    font=("Arial", 19, "bold")
).pack(pady=18)

# Information card function
def create_card(title, value, color):

    card = ctk.CTkFrame(
        sidebar, fg_color="#1B2942", corner_radius=12
    )
    card.pack(fill="x", padx=15, pady=7)

    ctk.CTkLabel(
        card,
        text=title,
        font=("Arial", 12),
        text_color="#94A3B8"
    ).pack(pady=(10, 3))

    label = ctk.CTkLabel(
        card,
        text=value,
        font=("Arial", 23, "bold"),
        text_color=color
    )
    label.pack(pady=(2, 12))

    return label


status_label = create_card(
    "DETECTION STATUS", "READY", "#22C55E"
)

count_label = create_card(
    "TOTAL OBJECTS", "0", "#38BDF8"
)

fps_label = create_card(
    "PROCESSING SPEED", "0 FPS", "#FACC15"
)

record_label = create_card(
    "RECORDING STATUS", "OFF", "#EF4444"
)

# Object details
ctk.CTkLabel(
    sidebar,
    text="DETECTED OBJECTS",
    font=("Arial", 15, "bold")
).pack(pady=(15, 8))

object_details = ctk.CTkTextbox(
    sidebar,
    height=150,
    fg_color="#080F1D",
    text_color="#FACC15",
    font=("Arial", 14),
    corner_radius=10
)
object_details.pack(
    fill="x", padx=15, pady=5
)
object_details.insert("1.0", "Waiting for detection...")
object_details.configure(state="disabled")
# Detection History

ctk.CTkLabel(
    sidebar,
    text="DETECTION HISTORY",
    font=("Arial", 15, "bold")
).pack(pady=(12, 8))

history_box = ctk.CTkTextbox(
    sidebar,
    height=120,
    fg_color="#080F1D",
    text_color="#38BDF8",
    font=("Arial", 12),
    corner_radius=10
)

history_box.pack(
    fill="x", padx=15, pady=5
)

history_box.insert("1.0", "No detection history yet...")
history_box.configure(state="disabled")

# ---------- CAMERA FUNCTIONS ----------
def add_detection_history(name, confidence, track_id=None):

    global detection_history, seen_track_ids, last_history_time

    current_time = time.time()

    # Avoid repeatedly recording the same tracked object
    if track_id is not None:
        if track_id in seen_track_ids:
            return

        seen_track_ids.add(track_id)

    else:
        # Fallback when tracking ID is unavailable
        if current_time - last_history_time.get(name, 0) < 3:
            return

        last_history_time[name] = current_time

    timestamp = datetime.now().strftime("%H:%M:%S")

    entry = (
        f"{timestamp} | {name} | {confidence * 100:.1f}%"
    )

    detection_history.append(entry)

    # Keep only the latest 20 entries
    detection_history = detection_history[-20:]

    history_box.configure(state="normal")
    history_box.delete("1.0", "end")
    history_box.insert(
        "1.0",
        "\n".join(reversed(detection_history))
    )
    history_box.configure(state="disabled")

def start_detection():
    print("START BUTTON CLICKED")
    global cap, running, prev_time

    if running:
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    print("Camera opened:", cap.isOpened())
    

    if not cap.isOpened():
        cap.release()
        cap = None
        messagebox.showerror("Camera Error", "Webcam could not be opened")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ret, test_frame = cap.read()
    print("Frame received:", ret)

    if not ret:
        cap.release()
        cap = None
        messagebox.showerror("Camera Error", "Could not read webcam frames")
        return

    running = True
    prev_time = time.time()

    status_label.configure(
        text="DETECTING",
        text_color="#22C55E"
    )

    update_frame()



def update_frame():

    print("UPDATE FRAME RUNNING")

    global prev_time, latest_frame

    if not running or cap is None:
        return

    ret, frame = cap.read()

    if not ret:
        stop_detection()
        return

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=416,
        conf=0.35,
        verbose=False
    )

    output = results[0].plot()

    # Save latest annotated frame for screenshot
    latest_frame = output.copy()

    # Object counts
    
# Object counts and confidence scores

    boxes = results[0].boxes

    class_ids = boxes.cls.tolist()
    confidences = boxes.conf.tolist()

    names = [
        model.names[int(i)] for i in class_ids
    ]

    # Get tracking IDs
    if boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
    else:
        track_ids = [None] * len(names)

    # Add detections to history
    for name, confidence, track_id in zip(
        names, confidences, track_ids
    ):
        add_detection_history(
            name,
            confidence,
            track_id
        )

    counts = Counter(names)

    count_label.configure(text=str(len(names)))

    # Prepare confidence details
    details_list = []

    for name, confidence in zip(names, confidences):
        details_list.append(
            f"{name}: {confidence * 100:.1f}%"
        )

    details = (
        "\n".join(details_list)
        if details_list else "No objects detected"
    )

    object_details.configure(state="normal")
    object_details.delete("1.0", "end")
    object_details.insert("1.0", details)
    object_details.configure(state="disabled")

    # FPS
    current_time = time.time()
    fps = 1 / max(current_time - prev_time, 0.001)
    prev_time = current_time

    fps_label.configure(text=f"{fps:.1f} FPS")

    # Recording
    if recording and out is not None:
        out.write(output)

    # Display camera
    rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image = image.resize((720, 480))

    photo = ImageTk.PhotoImage(image)

    camera_screen.configure(image=photo, text="")
    camera_screen.image = photo

    root.after(10, update_frame)


def start_recording():

    global out, recording

    if not running:
        messagebox.showwarning(
            "Warning", "Start detection first"
        )
        return

    if recording:
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        "gui_detection.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (width, height)
    )

    if not out.isOpened():
        out.release()
        out = None
        messagebox.showerror(
            "Recording Error", "Could not start recording"
        )
        return

    recording = True

    record_label.configure(
        text="RECORDING", text_color="#22C55E"
    )


def stop_recording():

    global out, recording

    if out is not None:
        out.release()
        out = None

    recording = False

    record_label.configure(
        text="OFF", text_color="#EF4444"
    )


def stop_detection():

    global cap, running, latest_frame

    running = False
    stop_recording()

    if cap is not None:
        cap.release()
        cap = None

    latest_frame = None
    detection_history = []
    seen_track_ids = set()
    last_history_time = {}

    camera_screen.configure(
        image="", text="CAMERA STOPPED"
    )
    camera_screen.image = None

    status_label.configure(
        text="STOPPED", text_color="#EF4444"
    )


# ---------- SCREENSHOT FUNCTION ----------

def take_screenshot():

    if not running or latest_frame is None:
        messagebox.showwarning(
            "Camera Not Active",
            "Please start detection before taking a screenshot."
        )
        return

    os.makedirs("screenshots", exist_ok=True)

    filename = datetime.now().strftime(
        "screenshots/detection_%Y%m%d_%H%M%S.jpg"
    )

    success = cv2.imwrite(filename, latest_frame)

    if success:
        messagebox.showinfo(
            "Screenshot Saved",
            f"Screenshot saved successfully:\n{filename}"
        )
    else:
        messagebox.showerror(
            "Screenshot Error",
            "Could not save screenshot."
        )

def export_detection_report():

    if not detection_history:
        messagebox.showwarning(
            "No Data",
            "No detection history available to export."
        )
        return

    os.makedirs("reports", exist_ok=True)

    filename = datetime.now().strftime(
        "reports/detection_report_%Y%m%d_%H%M%S.csv"
    )

    try:
        with open(filename, "w", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)

            # CSV headings
            writer.writerow([
                "Timestamp",
                "Object",
                "Confidence"
            ])

            # Export history
            for entry in detection_history:

                timestamp, name, confidence = entry.split(" | ")

                writer.writerow([
                    timestamp,
                    name,
                    confidence
                ])

        messagebox.showinfo(
            "Report Exported",
            f"Detection report saved successfully:\n{filename}"
        )

    except Exception as e:
        messagebox.showerror(
            "Export Error",
            f"Could not export report:\n{e}"
        )


def close_app():

    stop_detection()
    root.destroy()


# ---------- CONTROL PANEL ----------

controls = ctk.CTkFrame(
    root, fg_color="#111C32", corner_radius=15
)
controls.pack(fill="x", padx=20, pady=(5, 15))

ctk.CTkButton(
    controls,
    text="START DETECTION",
    fg_color="#16A34A",
    hover_color="#15803D",
    width=175,
    height=42,
    command=start_detection
).pack(side="left", padx=15, pady=15)

ctk.CTkButton(
    controls,
    text="STOP DETECTION",
    fg_color="#DC2626",
    hover_color="#B91C1C",
    width=175,
    height=42,
    command=stop_detection
).pack(side="left", padx=10)

ctk.CTkButton(
    controls,
    text="TAKE SCREENSHOT",
    fg_color="#16A34A",
    hover_color="#15803D",
    width=175,
    height=42,
    command=take_screenshot
).pack(side="left", padx=10)

ctk.CTkButton(
    controls,
    text="EXPORT CSV REPORT",
    fg_color="#9333EA",
    hover_color="#7E22CE",
    width=175,
    height=42,
    command=export_detection_report
).pack(side="left", padx=10)

ctk.CTkButton(
    controls,
    text="START RECORDING",
    fg_color="#2563EB",
    hover_color="#1D4ED8",
    width=175,
    height=42,
    command=start_recording
).pack(side="left", padx=10)

ctk.CTkButton(
    controls,
    text="STOP RECORDING",
    fg_color="#D97706",
    hover_color="#B45309",
    width=175,
    height=42,
    command=stop_recording
).pack(side="left", padx=10)

ctk.CTkButton(
    controls,
    text="EXIT",
    fg_color="#475569",
    hover_color="#334155",
    width=100,
    height=42,
    command=close_app
).pack(side="right", padx=15)

root.protocol("WM_DELETE_WINDOW", close_app)

root.mainloop()
