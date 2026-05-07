import tkinter as tk

from tkinter import ttk

from math import atan, degrees

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Constants

MOON_ANGULAR_DIAMETER = 1860  # Arcseconds (average)


# Function to calculate and update angular diameter and relative size

def update_calculations():

    try:

        # Get slider values

        size = size_slider.get()  # Object size in meters

        distance = distance_slider.get()  # Distance in kilometers


        # Convert distance to meters

        distance_m = distance * 1000


        # Calculate angular diameter

        angular_diameter_rad = 2 * atan((size / 2) / distance_m)

        angular_diameter_arcsec = angular_diameter_rad * 206265


        # Calculate relative size compared to the Moon

        relative_size = (angular_diameter_arcsec / MOON_ANGULAR_DIAMETER) * 100


        # Update labels

        angular_diameter_label.config(text=f"Angular Diameter: {angular_diameter_arcsec:.2f} arcseconds")

        relative_size_label.config(text=f"Relative to Moon: {relative_size:.2f}%")


        # Update virtual sky

        update_virtual_sky(angular_diameter_arcsec, relative_size)

    except Exception as e:

        angular_diameter_label.config(text="Error in calculation")

        relative_size_label.config(text=str(e))


# Function to update virtual sky visualization

def update_virtual_sky(angular_diameter, relative_size):

    ax.clear()

    ax.set_title("Virtual Sky")

    ax.set_xlim(-1, 1)

    ax.set_ylim(-1, 1)


    # Plot the Moon as the reference object

    moon_radius = 0.5

    moon_circle = plt.Circle((0, 0), moon_radius, color='gray', alpha=0.5, label="Moon")

    ax.add_artist(moon_circle)


    # Plot the object as a scaled circle

    object_radius = (angular_diameter / MOON_ANGULAR_DIAMETER) * moon_radius

    object_circle = plt.Circle((0.6, 0), object_radius, color='blue', alpha=0.7, label="Object")

    ax.add_artist(object_circle)


    # Add legend

    ax.legend()

    canvas.draw()


# GUI Setup

root = tk.Tk()

root.title("Satellite Angular Diameter Visualizer")


# Sliders for size and distance

tk.Label(root, text="Object Size (meters):").grid(row=0, column=0, padx=5, pady=5, sticky="w")

size_slider = tk.Scale(root, from_=1, to_=1000, resolution=0.1, orient="horizontal", length=300, command=lambda x: update_calculations())

size_slider.set(10)  # Default value

size_slider.grid(row=0, column=1, padx=5, pady=5)


tk.Label(root, text="Distance (kilometers):").grid(row=1, column=0, padx=5, pady=5, sticky="w")

distance_slider = tk.Scale(root, from_=100, to_=40000, resolution=100, orient="horizontal", length=300, command=lambda x: update_calculations())

distance_slider.set(500)  # Default value

distance_slider.grid(row=1, column=1, padx=5, pady=5)


# Output labels

angular_diameter_label = tk.Label(root, text="Angular Diameter: -- arcseconds")

angular_diameter_label.grid(row=2, column=0, columnspan=2, pady=10)


relative_size_label = tk.Label(root, text="Relative to Moon: --%")

relative_size_label.grid(row=3, column=0, columnspan=2, pady=10)


# Matplotlib Figure for Virtual Sky

fig, ax = plt.subplots(figsize=(5, 5))

canvas = FigureCanvasTkAgg(fig, master=root)

canvas_widget = canvas.get_tk_widget()

canvas_widget.grid(row=4, column=0, columnspan=2, pady=10)


# Initial Calculation

update_calculations()


# Start GUI

root.mainloop()