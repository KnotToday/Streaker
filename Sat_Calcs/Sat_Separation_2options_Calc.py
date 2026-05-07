import tkinter as tk
from tkinter import messagebox
from datetime import datetime
from math import cos, sin, radians, sqrt
from skyfield.api import Loader, EarthSatellite


# Function to convert Stellarium RA/Dec to decimal degrees
def convert_ra_dec(ra, dec):
    """Convert RA/Dec in Stellarium format to decimal degrees."""
    try:
        # Parse RA (hh:mm:ss -> degrees)
        ra = ra.replace('h', ':').replace('m', ':').replace('s', ':').replace(' ', ':').strip(':')
        ra_parts = ra.split(':')
        ra_hours = float(ra_parts[0])
        ra_minutes = float(ra_parts[1])
        ra_seconds = float(ra_parts[2])
        ra_decimal = (ra_hours + ra_minutes / 60 + ra_seconds / 3600) * 15

        # Parse Dec (±dd:mm:ss -> degrees)
        dec = dec.replace('°', ':').replace('\'', ':').replace('"', ':').replace(' ', ':').strip(':')
        dec_sign = -1 if '-' in dec else 1
        dec_parts = dec.lstrip('-+').split(':')
        dec_degrees = float(dec_parts[0])
        dec_minutes = float(dec_parts[1])
        dec_seconds = float(dec_parts[2])
        dec_decimal = dec_sign * (dec_degrees + dec_minutes / 60 + dec_seconds / 3600)

        return ra_decimal, dec_decimal
    except Exception as e:
        raise ValueError(f"Invalid RA/Dec format: {e}")


# Function to calculate separation
def calculate_separation():
    try:
        if use_tle.get():
            # TLE-based calculation
            tle1_line1 = tle1_line1_entry.get().strip()
            tle1_line2 = tle1_line2_entry.get().strip()
            tle2_line1 = tle2_line1_entry.get().strip()
            tle2_line2 = tle2_line2_entry.get().strip()

            # Load Skyfield resources
            load = Loader('./skyfield-data')
            ts = load.timescale()

            # Define satellites
            satellite1 = EarthSatellite(tle1_line1, tle1_line2, "Satellite 1", ts)
            satellite2 = EarthSatellite(tle2_line1, tle2_line2, "Satellite 2", ts)

            # Get observation time
            julian_day = float(julian_day_entry.get())
            fractional_day = float(fractional_day_entry.get())
            time = ts.tt_jd(julian_day + fractional_day)

            # Compute positions
            pos1 = satellite1.at(time).position.km
            pos2 = satellite2.at(time).position.km
        elif use_ra_dec.get():
            # RA/Dec-based calculation
            ra1_raw = ra1_entry.get()
            dec1_raw = dec1_entry.get()
            dist1 = float(dist1_entry.get())
            ra2_raw = ra2_entry.get()
            dec2_raw = dec2_entry.get()
            dist2 = float(dist2_entry.get())

            # Convert RA/Dec
            ra1, dec1 = convert_ra_dec(ra1_raw, dec1_raw)
            ra2, dec2 = convert_ra_dec(ra2_raw, dec2_raw)

            # Convert RA/Dec to Cartesian coordinates
            pos1 = ra_dec_to_cartesian(ra1, dec1, dist1)
            pos2 = ra_dec_to_cartesian(ra2, dec2, dist2)
        else:
            # Stellarium Julian Date-based calculation
            julian_date = float(stellarium_julian_entry.get())
            load = Loader('./skyfield-data')
            ts = load.timescale()
            time = ts.tt_jd(julian_date)

            # TLE satellites must still be provided
            tle1_line1 = tle1_line1_entry.get().strip()
            tle1_line2 = tle1_line2_entry.get().strip()
            tle2_line1 = tle2_line1_entry.get().strip()
            tle2_line2 = tle2_line2_entry.get().strip()

            satellite1 = EarthSatellite(tle1_line1, tle1_line2, "Satellite 1", ts)
            satellite2 = EarthSatellite(tle2_line1, tle2_line2, "Satellite 2", ts)

            # Compute positions
            pos1 = satellite1.at(time).position.km
            pos2 = satellite2.at(time).position.km

        # Calculate Euclidean separation distance
        separation = sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

        # Display result
        result_label.config(text=f"Separation Distance: {separation:.2f} km")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Function to calculate Cartesian coordinates from RA/Dec
def ra_dec_to_cartesian(ra, dec, distance):
    ra_rad = radians(ra)
    dec_rad = radians(dec)
    x = distance * cos(dec_rad) * cos(ra_rad)
    y = distance * cos(dec_rad) * sin(ra_rad)
    z = distance * sin(dec_rad)
    return x, y, z


# GUI Setup
root = tk.Tk()
root.title("Satellite Separation Calculator")

# Mode Toggles
use_tle = tk.BooleanVar(value=True)
use_ra_dec = tk.BooleanVar(value=False)
use_stellarium = tk.BooleanVar(value=False)

tk.Checkbutton(root, text="Use TLE Data", variable=use_tle).grid(row=0, column=0, sticky="w")
tk.Checkbutton(root, text="Use RA/Dec Data", variable=use_ra_dec).grid(row=1, column=0, sticky="w")
tk.Checkbutton(root, text="Use Stellarium Julian Date", variable=use_stellarium).grid(row=2, column=0, sticky="w")

# TLE Inputs
tle1_line1_entry = tk.Entry(root, width=60)
tle1_line2_entry = tk.Entry(root, width=60)
tle2_line1_entry = tk.Entry(root, width=60)
tle2_line2_entry = tk.Entry(root, width=60)

tk.Label(root, text="Satellite 1 TLE Line 1:").grid(row=3, column=0, sticky="e")
tle1_line1_entry.grid(row=3, column=1)
tk.Label(root, text="Satellite 1 TLE Line 2:").grid(row=4, column=0, sticky="e")
tle1_line2_entry.grid(row=4, column=1)
tk.Label(root, text="Satellite 2 TLE Line 1:").grid(row=5, column=0, sticky="e")
tle2_line1_entry.grid(row=5, column=1)
tk.Label(root, text="Satellite 2 TLE Line 2:").grid(row=6, column=0, sticky="e")
tle2_line2_entry.grid(row=6, column=1)

# RA/Dec Inputs
ra1_entry = tk.Entry(root, width=20)
dec1_entry = tk.Entry(root, width=20)
dist1_entry = tk.Entry(root, width=10)
ra2_entry = tk.Entry(root, width=20)
dec2_entry = tk.Entry(root, width=20)
dist2_entry = tk.Entry(root, width=10)

tk.Label(root, text="Satellite 1 RA (hh:mm:ss):").grid(row=7, column=0, sticky="e")
ra1_entry.grid(row=7, column=1)
tk.Label(root, text="Satellite 1 Dec (±dd:mm:ss):").grid(row=8, column=0, sticky="e")
dec1_entry.grid(row=8, column=1)
tk.Label(root, text="Satellite 1 Distance (km):").grid(row=9, column=0, sticky="e")
dist1_entry.grid(row=9, column=1)

tk.Label(root, text="Satellite 2 RA (hh:mm:ss):").grid(row=10, column=0, sticky="e")
ra2_entry.grid(row=10, column=1)
tk.Label(root, text="Satellite 2 Dec (±dd:mm:ss):").grid(row=11, column=0, sticky="e")
dec2_entry.grid(row=11, column=1)
tk.Label(root, text="Satellite 2 Distance (km):").grid(row=12, column=0, sticky="e")
dist2_entry.grid(row=12, column=1)

# Stellarium Julian Date Input
stellarium_julian_entry = tk.Entry(root, width=20)
tk.Label(root, text="Stellarium Julian Date:").grid(row=13, column=0, sticky="e")
stellarium_julian_entry.grid(row=13, column=1)

# Julian Date Inputs
julian_day_entry = tk.Entry(root, width=20)
fractional_day_entry = tk.Entry(root, width=20)
tk.Label(root, text="Julian Day:").grid(row=14, column=0, sticky="e")
julian_day_entry.grid(row=14, column=1)
tk.Label(root, text="Fractional Day:").grid(row=15, column=0, sticky="e")
fractional_day_entry.grid(row=15, column=1)

# Calculate Separation Button
calculate_button = tk.Button(root, text="Calculate Separation", command=calculate_separation)
calculate_button.grid(row=16, column=0, columnspan=2, pady=10)

# Result Display
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.grid(row=17, column=0, columnspan=2, pady=10)

# Start the GUI
root.mainloop()
