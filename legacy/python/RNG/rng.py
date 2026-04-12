import tkinter as tk
import random
from PIL import Image, ImageTk

def generate_random_number():
    random_number = random.randint(1, 100)
    result_label.config(text=f"{random_number}")

# Create the main window
root = tk.Tk()
root.title("Random Number Generator")
root.geometry("600x250")

# Set the window icon
root.iconbitmap("white.ico")

# Load and resize the image
image = Image.open("white.png")
image = image.resize((100, 100), Image.LANCZOS)
photo = ImageTk.PhotoImage(image)

# Create a frame to hold the image and text
frame = tk.Frame(root)
frame.pack(pady=10)

# Create a label for the image
image_label = tk.Label(frame, image=photo)
image_label.pack(side="left", padx=10)

# Create a label for the text
label = tk.Label(frame, text="Generate a random number between 1 and 100", font=("Helvetica", 14))
label.pack(side="left", padx=10)

# Create a result label
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=(0, 10))

# Create a button
generate_button = tk.Button(root, text="Generate Random Number", font=("Helvetica", 14), command=generate_random_number)
generate_button.pack(pady=10)

# Run the application
root.mainloop()
