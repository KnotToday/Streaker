import tkinter as tk
from tkinter import ttk, messagebox
from math import cos, sin, radians, sqrt
from skyfield.api import Loader, EarthSatellite

# Constants
EARTH_RADIUS = 6371  # Earth's radius in kilometers

# Function to convert Stellarium RA/Dec to decimal degrees
def convert_ra_dec(ra, dec):
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


# Function to calculate separation for RA/Dec data
def calculate_ra_dec_separation():
    try:
        # Get RA/Dec inputs
        ra1_raw = ra1_entry.get()
        dec1_raw = dec1_entry.get()
        alt_or_range1 = alt_range1_var.get()
        alt1 = altitude1_entry.get()
        range1 = range1_entry.get()

        ra2_raw = ra2_entry.get()
        dec2_raw = dec2_entry.get()
        alt_or_range2 = alt_range2_var.get()
        alt2 = altitude2_entry.get()
        range2 = range2_entry.get()

        if not (ra1_raw and dec1_raw and ra2_raw and dec2_raw):
            raise ValueError("RA/Dec data is incomplete. Please fill all fields.")

        # Convert Altitude to Range if necessary
        if alt_or_range1 == "Altitude":
            range1 = EARTH_RADIUS + float(alt1)
        else:
            range1 = float(range1)

        if alt_or_range2 == "Altitude":
            range2 = EARTH_RADIUS + float(alt2)
        else:
            range2 = float(range2)

        # Debugging: Print range values
        print(f"Satellite 1 Range: {range1} km")
        print(f"Satellite 2 Range: {range2} km")

        # Convert RA/Dec
        ra1, dec1 = convert_ra_dec(ra1_raw, dec1_raw)
        ra2, dec2 = convert_ra_dec(ra2_raw, dec2_raw)

        # Convert RA/Dec to Cartesian coordinates
        pos1 = ra_dec_to_cartesian(ra1, dec1, range1)
        pos2 = ra_dec_to_cartesian(ra2, dec2, range2)

        # Debugging: Print positions
        print(f"RA/Dec Satellite 1 Position: {pos1}")
        print(f"RA/Dec Satellite 2 Position: {pos2}")

        # Calculate Euclidean separation distance
        separation = sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

        # Display result
        result_label.config(text=f"Separation Distance: {separation:.2f} km")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error details: {e}")


# Function to calculate separation for TLE data
def calculate_tle_separation():
    try:
        # Get TLE inputs
        tle1_line1 = tle1_line1_entry.get().strip()
        tle1_line2 = tle1_line2_entry.get().strip()
        tle2_line1 = tle2_line1_entry.get().strip()
        tle2_line2 = tle2_line2_entry.get().strip()

        if not (tle1_line1 and tle1_line2 and tle2_line1 and tle2_line2):
            raise ValueError("TLE data is incomplete. Please fill all TLE fields.")

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

        # Validate time relative to TLE epoch
        max_offset_days = 30  # Allow up to 30 days difference
        epoch1_offset = abs((time.tt - satellite1.epoch.tt))
        epoch2_offset = abs((time.tt - satellite2.epoch.tt))

        if epoch1_offset > max_offset_days or epoch2_offset > max_offset_days:
            raise ValueError(
                f"Julian date is too far from TLE epoch.\n"
                f"Satellite 1 offset: {epoch1_offset:.2f} days\n"
                f"Satellite 2 offset: {epoch2_offset:.2f} days"
            )

        # Compute positions
        pos1 = satellite1.at(time).position.km
        pos2 = satellite2.at(time).position.km

        # Debugging: Print positions
        print(f"TLE Satellite 1 Position: {pos1}")
        print(f"TLE Satellite 2 Position: {pos2}")

        # Calculate Euclidean separation distance
        separation = sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

        # Display result
        result_label.config(text=f"Separation Distance: {separation:.2f} km")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error details: {e}")


# Function to calculate Cartesian coordinates from RA/Dec
def ra_dec_to_cartesian(ra, dec, distance):
    ra_rad = radians(ra)
    dec_rad = radians(dec)
    x = distance * cos(dec_rad) * cos(ra_rad)
    y = distance * cos(dec_rad) * sin(ra_rad)
    z = distance * sin(dec_rad)

    # Debugging: Print intermediate values
    print(f"RA (deg): {ra}, Dec (deg): {dec}, Distance (km): {distance}")
    print(f"Converted Cartesian Coordinates: x={x}, y={y}, z={z}")

    return x, y, z


# GUI Setup
root = tk.Tk()
root.title("Satellite Separation Calculator")

# Tabs
tabs = ttk.Notebook(root)

# TLE Tab
tle_tab = ttk.Frame(tabs)
tabs.add(tle_tab, text="TLE Data")

# TLE Inputs
tk.Label(tle_tab, text="Satellite 1 TLE Line 1:").grid(row=0, column=0, sticky="e")
tle1_line1_entry = tk.Entry(tle_tab, width=60)
tle1_line1_entry.grid(row=0, column=1)

tk.Label(tle_tab, text="Satellite 1 TLE Line 2:").grid(row=1, column=0, sticky="e")
tle1_line2_entry = tk.Entry(tle_tab, width=60)
tle1_line2_entry.grid(row=1, column=1)

tk.Label(tle_tab, text="Satellite 2 TLE Line 1:").grid(row=2, column=0, sticky="e")
tle2_line1_entry = tk.Entry(tle_tab, width=60)
tle2_line1_entry.grid(row=2, column=1)

tk.Label(tle_tab, text="Satellite 2 TLE Line 2:").grid(row=3, column=0, sticky="e")
tle2_line2_entry = tk.Entry(tle_tab, width=60)
tle2_line2_entry.grid(row=3, column=1)

tk.Label(tle_tab, text="Julian Day:").grid(row=4, column=0, sticky="e")
julian_day_entry = tk.Entry(tle_tab, width=20)
julian_day_entry.grid(row=4, column=1)

tk.Label(tle_tab, text="Fractional Day:").grid(row=5, column=0, sticky="e")
fractional_day_entry = tk.Entry(tle_tab, width=20)
fractional_day_entry.grid(row=5, column=1)

# TLE Calculation Button
tle_calculate_button = tk.Button(
    tle_tab, text="Calculate Separation", command=calculate_tle_separation
)
tle_calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

# RA/Dec Tab
ra_dec_tab = ttk.Frame(tabs)
tabs.add(ra_dec_tab, text="RA/Dec Data")

tabs.pack(expand=1, fill="both")

# RA/Dec Inputs
tk.Label(ra_dec_tab, text="Satellite 1 RA (hh:mm:ss):").grid(row=0, column=0, sticky="e")
ra1_entry = tk.Entry(ra_dec_tab, width=20)
ra1_entry.grid(row=0, column=1)

tk.Label(ra_dec_tab, text="Satellite 1 Dec (±dd:mm:ss):").grid(row=1, column=0, sticky="e")
dec1_entry = tk.Entry(ra_dec_tab, width=20)
dec1_entry.grid(row=1, column=1)

alt_range1_var = tk.StringVar(value="Range")
tk.Radiobutton(ra_dec_tab, text="Altitude", variable=alt_range1_var, value="Altitude").grid(row=2, column=0, sticky="w")
tk.Radiobutton(ra_dec_tab, text="Range", variable=alt_range1_var, value="Range").grid(row=2, column=1, sticky="w")

tk.Label(ra_dec_tab, text="Satellite 1 Altitude (km):").grid(row=3, column=0, sticky="e")
altitude1_entry = tk.Entry(ra_dec_tab, width=10)
altitude1_entry.grid(row=3, column=1)

tk.Label(ra_dec_tab, text="Satellite 1 Range (km):").grid(row=4, column=0, sticky="e")
range1_entry = tk.Entry(ra_dec_tab, width=10)
range1_entry.grid(row=4, column=1)

# Same setup for Satellite 2
tk.Label(ra_dec_tab, text="Satellite 2 RA (hh:mm:ss):").grid(row=5, column=0, sticky="e")
ra2_entry = tk.Entry(ra_dec_tab, width=20)
ra2_entry.grid(row=5, column=1)

tk.Label(ra_dec_tab, text="Satellite 2 Dec (±dd:mm:ss):").grid(row=6, column=0, sticky="e")
dec2_entry = tk.Entry(ra_dec_tab, width=20)
dec2_entry.grid(row=6, column=1)

alt_range2_var = tk.StringVar(value="Range")
tk.Radiobutton(ra_dec_tab, text="Altitude", variable=alt_range2_var, value="Altitude").grid(row=7, column=0, sticky="w")
tk.Radiobutton(ra_dec_tab, text="Range", variable=alt_range2_var, value="Range").grid(row=7, column=1, sticky="w")

tk.Label(ra_dec_tab, text="Satellite 2 Altitude (km):").grid(row=8, column=0, sticky="e")
altitude2_entry = tk.Entry(ra_dec_tab, width=10)
altitude2_entry.grid(row=8, column=1)

tk.Label(ra_dec_tab, text="Satellite 2 Range (km):").grid(row=9, column=0, sticky="e")
range2_entry = tk.Entry(ra_dec_tab, width=10)
range2_entry.grid(row=9, column=1)

# RA/Dec Calculation Button
ra_dec_calculate_button = tk.Button(ra_dec_tab, text="Calculate Separation", command=calculate_ra_dec_separation)
ra_dec_calculate_button.grid(row=10, column=0, columnspan=2, pady=10)

# Result Display
# Result Labels
tle_result_label = tk.Label(tle_tab, text="", font=("Arial", 14))
tle_result_label.grid(row=7, column=0, columnspan=2, pady=10)

ra_dec_result_label = tk.Label(ra_dec_tab, text="", font=("Arial", 14))
ra_dec_result_label.grid(row=11, column=0, columnspan=2, pady=10)

# Modify Calculation Functions
def calculate_tle_separation():
    try:
        # (Existing TLE calculation code...)

        # Update TLE result label
        tle_result_label.config(text=f"Separation Distance: {separation:.2f} km")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error details: {e}")

def calculate_ra_dec_separation():
    try:
        # (Existing RA/Dec calculation code...)

        # Update RA/Dec result label
        ra_dec_result_label.config(text=f"Separation Distance: {separation:.2f} km")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error details: {e}")

# Start the GUI
root.mainloop()
