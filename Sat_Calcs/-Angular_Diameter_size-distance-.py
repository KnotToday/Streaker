import tkinter as tk

from math import atan, degrees


def calculate_angular_diameter():

    try:

        # Get user inputs

        size = float(size_entry.get())  # Satellite size in meters

        distance = float(distance_entry.get())  # Distance in kilometers

        latitude = float(latitude_entry.get())  # Observer latitude

        longitude = float(longitude_entry.get())  # Observer longitude

        elevation = float(elevation_entry.get())  # Elevation in meters


        # Convert distance to meters

        distance_m = distance * 1000


        # Calculate angular diameter

        angular_diameter_rad = 2 * atan((size / 2) / distance_m)

        angular_diameter_arcsec = angular_diameter_rad * (206265)


        # Display result

        result_label.config(text=f"Angular Diameter: {angular_diameter_arcsec:.2f} arcseconds")

    except Exception as e:

        result_label.config(text=f"Error: {e}")


# GUI Setup

root = tk.Tk()

root.title("Satellite Angular Diameter Calculator")


# Inputs

tk.Label(root, text="Satellite Size (m):").grid(row=0, column=0, padx=5, pady=5, sticky="e")

size_entry = tk.Entry(root)

size_entry.grid(row=0, column=1, padx=5, pady=5)


tk.Label(root, text="Distance (km):").grid(row=1, column=0, padx=5, pady=5, sticky="e")

distance_entry = tk.Entry(root)

distance_entry.grid(row=1, column=1, padx=5, pady=5)


tk.Label(root, text="Observer Latitude (°):").grid(row=2, column=0, padx=5, pady=5, sticky="e")

latitude_entry = tk.Entry(root)

latitude_entry.grid(row=2, column=1, padx=5, pady=5)


tk.Label(root, text="Observer Longitude (°):").grid(row=3, column=0, padx=5, pady=5, sticky="e")

longitude_entry = tk.Entry(root)

longitude_entry.grid(row=3, column=1, padx=5, pady=5)


tk.Label(root, text="Elevation (m):").grid(row=4, column=0, padx=5, pady=5, sticky="e")

elevation_entry = tk.Entry(root)

elevation_entry.grid(row=4, column=1, padx=5, pady=5)


# Calculate Button

calculate_button = tk.Button(root, text="Calculate Angular Diameter", command=calculate_angular_diameter)

calculate_button.grid(row=5, column=0, columnspan=2, pady=10)


# Result Display

result_label = tk.Label(root, text="", font=("Arial", 14))

result_label.grid(row=6, column=0, columnspan=2, pady=10)


# Start GUI loop

root.mainloop()