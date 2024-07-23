import tkinter as tk
import random

def generate_random_number():
    random_number = random.randint(1, 100)
    result_label.config(text=f"{random_number}")

# Create the main window
root = tk.Tk()
root.title("Random Number Generator")
root.geometry("500x200")

# Create a label
label = tk.Label(root, text="Generate a random number between 1 and 100", font=("Helvetica", 14))
label.pack(pady=20)

# Create a result label
result_label = tk.Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=10)

# Create a button
generate_button = tk.Button(root, text="Generate Random Number", font=("Helvetica", 14), command=generate_random_number)
generate_button.pack(pady=10)

# Run the application
root.mainloop()
