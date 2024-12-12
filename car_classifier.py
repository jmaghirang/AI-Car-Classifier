# Import Libraries
import tkinter as tk
from tkinter import filedialog, Label, Button
from fastai.vision import *
from fastai.metrics import accuracy
from PIL import Image, ImageTk
import scipy.io
import json

# Define FocalLoss Class
class FocalLoss(Module):
    def __init__(self, gamma=2., alpha=0.5):
        self.gamma, self.alpha = gamma, alpha

    def forward(self, input, target):
        p = input.softmax(dim=-1)
        log_p = p.log()
        loss = -(self.alpha * (1 - p) ** self.gamma * log_p)
        return loss.gather(dim=-1, index=target.unsqueeze(-1)).mean()

# Load FastAI ResNet Model
model = load_learner('.', 'fastai_resnet.pkl')

# Extract Mapping from cars_annos.mat
def extract_label_mapping(mat_file):
    annotations = scipy.io.loadmat(mat_file)
    class_names = annotations['class_names'][0]
    return {str(i + 1).zfill(4): str(c[0]) for i, c in enumerate(class_names)}

# Load the mapping from cars_annos.mat
label_mapping = extract_label_mapping('cars_annos.mat')

# Save mapping to JSON 
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)

# Prediction Function
def predict(image_path):
    img = open_image(image_path)
    pred_class, pred_idx, outputs = model.predict(img)

    # Map Class Label to Car Name
    car_name = label_mapping.get(str(pred_class), "Unknown Model")
    
    prediction_text.set(f"""
    Car Model: {car_name}
    """)

# Upload and Prediction Logic
def upload_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if file_path:
        img = Image.open(file_path)
        img_resized = img.resize((400, 300))
        img_tk = ImageTk.PhotoImage(img_resized)
        image_label.config(image=img_tk)
        image_label.image = img_tk
        
        predict(file_path)

# GUI Setup
window = tk.Tk()
window.title("Car Classification - FastAI ResNet")
window.geometry("800x600")

# GUI Elements
title_label = Label(window, text="Car Classifier (FastAI ResNet)", font=("Arial", 24))
title_label.pack(pady=20)

upload_btn = Button(window, text="Upload Car Image", command=upload_image, font=("Arial", 14))
upload_btn.pack(pady=10)

image_label = Label(window)
image_label.pack(pady=10)

prediction_text = tk.StringVar()
prediction_label = Label(window, textvariable=prediction_text, font=("Arial", 14))
prediction_label.pack(pady=20)

# Start GUI Loop
window.mainloop()