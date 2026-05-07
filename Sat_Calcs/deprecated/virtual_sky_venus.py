import tkinter as tk

from math import atan

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Constants

MOON_ANGULAR_DIAMETER = 1860  # Arcseconds (average)

VENUS_DIAMETER_KM = 12104  # Venus's diameter in kilometers

VENUS_CLOSEST_DISTANCE_KM = 41000000  # Closest approach to Earth

VENUS_FARTHEST_DISTANCE_KM = 261000000  # Farthest distance from Earth


# Function to calculate and update angular diameter and relative size

def update_calculations():

    try:

        # Get values from entries

        size = float(size_entry.get())  # Object size in meters

        distance = float(distance_entry.get())  # Distance in kilometers


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


# Fine control for size and distance

def adjust_size(delta):

    size = float(size_entry.get())

    size = max(1, size + delta)  # Ensure size is at least 1 meter

    size_entry.delete(0, tk.END)

    size_entry.insert(0, f"{size:.1f}")

    update_calculations()


def adjust_distance(delta):

    distance = float(distance_entry.get())

    distance = max(100, distance + delta)  # Ensure distance is at least 100 km

    distance_entry.delete(0, tk.END)

    distance_entry.insert(0, f"{distance:.1f}")

    update_calculations()


# Function to set Venus as the object

def set_venus():

    size_entry.delete(0, tk.END)

    size_entry.insert(0, VENUS_DIAMETER_KM * 1000)  # Convert km to meters


    distance_entry.delete(0, tk.END)

    distance_entry.insert(0, VENUS_CLOSEST_DISTANCE_KM)  # Set default closest distance


    update_calculations()


# GUI Setup

root = tk.Tk()

root.title("Satellite Angular Diameter Visualizer")


# Object Size Controls

tk.Label(root, text="Object Size (meters):").grid(row=0, column=0, padx=5, pady=5, sticky="w")

size_entry = tk.Entry(root, width=10)

size_entry.grid(row=0, column=1, padx=5, pady=5)

size_entry.insert(0, "10")


tk.Button(root, text="–", command=lambda: adjust_size(-0.1)).grid(row=0, column=2, padx=5, pady=5)

tk.Button(root, text="+", command=lambda: adjust_size(0.1)).grid(row=0, column=3, padx=5, pady=5)


# Distance Controls

tk.Label(root, text="Distance (kilometers):").grid(row=1, column=0, padx=5, pady=5, sticky="w")

distance_entry = tk.Entry(root, width=10)

distance_entry.grid(row=1, column=1, padx=5, pady=5)

distance_entry.insert(0, "500")


tk.Button(root, text="–", command=lambda: adjust_distance(-1)).grid(row=1, column=2, padx=5, pady=5)

tk.Button(root, text="+", command=lambda: adjust_distance(1)).grid(row=1, column=3, padx=5, pady=5)


# Venus Preset Button

tk.Button(root, text="Set Venus", command=set_venus).grid(row=2, column=0, columnspan=4, pady=10)


# Output labels

angular_diameter_label = tk.Label(root, text="Angular Diameter: -- arcseconds")

angular_diameter_label.grid(row=3, column=0, columnspan=4, pady=10)


relative_size_label = tk.Label(root, text="Relative to Moon: --%")

relative_size_label.grid(row=4, column=0, columnspan=4, pady=10)


# Matplotlib Figure for Virtual Sky

fig, ax = plt.subplots(figsize=(5, 5))

canvas = FigureCanvasTkAgg(fig, master=root)

canvas_widget = canvas.get_tk_widget()

canvas_widget.grid(row=5, column=0, columnspan=4, pady=10)


# Initial Calculation

update_calculations()


# Start GUI

root.mainloop()