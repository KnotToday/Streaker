

import tkinter as tk

from math import atan

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Constants

MOON_DEFAULT_DIAMETER = 3474  # Moon's diameter in kilometers

MOON_DEFAULT_DISTANCE = 384400  # Average distance from Earth to Moon in kilometers

VENUS_DIAMETER = 12104  # Venus's diameter in kilometers

VENUS_DEFAULT_DISTANCE = 41000000  # Default Venus distance in kilometers


# Function to calculate angular diameter

def calculate_angular_diameter(size, distance):

    distance_m = distance * 1000  # Convert distance to meters

    angular_diameter_rad = 2 * atan((size / 2) / distance_m)

    return angular_diameter_rad * 206265  # Convert to arcseconds


# Function to update the virtual sky visualization

def update_virtual_sky():

    ax.clear()

    ax.set_title("Virtual Sky", fontsize=20, fontweight="bold")

    ax.set_xlim(-1, 1)

    ax.set_ylim(-1, 1)


    # Get Moon size and calculate its angular diameter

    moon_diameter = float(moon_size_entry.get())

    moon_angular_diameter = calculate_angular_diameter(moon_diameter, MOON_DEFAULT_DISTANCE)


    # Plot the Moon as the reference object

    moon_radius = 0.5  # Fixed visual size for the Moon

    moon_circle = plt.Circle((0, 0), moon_radius, color='gray', alpha=0.5, label="Moon")

    ax.add_artist(moon_circle)


    # Plot Venus (size depends on distance and Moon's size)

    venus_distance = float(venus_distance_entry.get())

    venus_angular_diameter = calculate_angular_diameter(VENUS_DIAMETER, venus_distance)

    venus_radius = (venus_angular_diameter / moon_angular_diameter) * moon_radius

    venus_circle = plt.Circle((0.6, 0), venus_radius, color='blue', alpha=0.7, label="Venus")

    ax.add_artist(venus_circle)


    # Plot UFO (calculated independently)

    object_size = float(object_size_entry.get())  # UFO size in meters

    object_distance = float(object_distance_entry.get())  # UFO distance in kilometers

    ufo_angular_diameter = calculate_angular_diameter(object_size, object_distance)

    ufo_radius = (ufo_angular_diameter / moon_angular_diameter) * moon_radius


    # Ensure UFO's radius is visible but realistically small

    ufo_radius = max(ufo_radius, 0.01)  # Minimum visible radius

    ufo_circle = plt.Circle((-0.6, 0), ufo_radius, color='green', alpha=0.7, label="UFO")

    ax.add_artist(ufo_circle)


    # Add legend

    ax.legend(fontsize=12)

    canvas.draw()


# Function to update all calculations and refresh the visualization

def update_calculations():

    update_virtual_sky()


# GUI Setup

root = tk.Tk()

root.title("Virtual Sky: Moon, Venus, and UFO")


# Make the window full-screen

root.attributes("-fullscreen", True)


# Set a common font

font_large = ("Arial", 16, "bold")


# Moon Controls

tk.Label(root, text="Moon Diameter (km):", font=font_large).grid(row=0, column=0, padx=10, pady=10, sticky="w")

moon_size_entry = tk.Entry(root, width=10, font=font_large)

moon_size_entry.grid(row=0, column=1, padx=10, pady=10)

moon_size_entry.insert(0, str(MOON_DEFAULT_DIAMETER))

moon_size_entry.bind("<Return>", lambda event: update_calculations())


# Venus Controls

tk.Label(root, text="Venus Distance (km):", font=font_large).grid(row=1, column=0, padx=10, pady=10, sticky="w")

venus_distance_entry = tk.Entry(root, width=10, font=font_large)

venus_distance_entry.grid(row=1, column=1, padx=10, pady=10)

venus_distance_entry.insert(0, str(VENUS_DEFAULT_DISTANCE))

venus_distance_entry.bind("<Return>", lambda event: update_calculations())


# UFO Controls

tk.Label(root, text="Object Size (meters):", font=font_large).grid(row=2, column=0, padx=10, pady=10, sticky="w")

object_size_entry = tk.Entry(root, width=10, font=font_large)

object_size_entry.grid(row=2, column=1, padx=10, pady=10)

object_size_entry.insert(0, "1")

object_size_entry.bind("<Return>", lambda event: update_calculations())


tk.Label(root, text="Object Distance (km):", font=font_large).grid(row=3, column=0, padx=10, pady=10, sticky="w")

object_distance_entry = tk.Entry(root, width=10, font=font_large)

object_distance_entry.grid(row=3, column=1, padx=10, pady=10)

object_distance_entry.insert(0, "500")

object_distance_entry.bind("<Return>", lambda event: update_calculations())


# Matplotlib Figure for Virtual Sky

fig, ax = plt.subplots(figsize=(8, 8))

canvas = FigureCanvasTkAgg(fig, master=root)

canvas_widget = canvas.get_tk_widget()

canvas_widget.grid(row=4, column=0, columnspan=2, pady=20)


# Initial Calculation and Visualization

update_virtual_sky()


# Start GUI

root.mainloop()
