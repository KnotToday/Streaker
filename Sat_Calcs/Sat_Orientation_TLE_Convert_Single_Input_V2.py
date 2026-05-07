import tkinter as tk
from tkinter import ttk, messagebox
from skyfield.api import Loader, EarthSatellite
from math import sqrt

# Constants
EARTH_RADIUS = 6371  # Earth's radius in kilometers

# Function to parse TLE data
def parse_tle_input(tle_text):
    """Parse TLE input text into individual satellite TLEs."""
    try:
        lines = tle_text.strip().splitlines()
        satellites = []
        current_sat = []
        for line in lines:
            if line.startswith('0 ') or line.startswith('1 ') or line.startswith('2 '):
                current_sat.append(line.strip())
                if len(current_sat) == 3:  # Each satellite has 3 lines: name, line 1, line 2
                    satellites.append(current_sat)
                    current_sat = []

        if len(satellites) != 2:
            raise ValueError("Expected TLE data for exactly 2 satellites.")

        return satellites[0][1], satellites[0][2], satellites[1][1], satellites[1][2]
    except Exception as e:
        raise ValueError(f"Error parsing TLE data: {e}")


# Function to format RA/Dec in Stellarium format
def format_ra_dec(ra, dec):
    """Convert RA/Dec from decimal to Stellarium's format."""
    # Convert RA from decimal hours to hh:mm:ss
    ra_hours = int(ra)
    ra_minutes = int((ra - ra_hours) * 60)
    ra_seconds = ((ra - ra_hours) * 60 - ra_minutes) * 60
    ra_formatted = f"{ra_hours}h{ra_minutes}m{ra_seconds:.2f}s"

    # Convert Dec from decimal degrees to ±dd:mm:ss
    dec_sign = '-' if dec < 0 else '+'
    dec = abs(dec)
    dec_degrees = int(dec)
    dec_minutes = int((dec - dec_degrees) * 60)
    dec_seconds = ((dec - dec_degrees) * 60 - dec_minutes) * 60
    dec_formatted = f"{dec_sign}{dec_degrees}°{dec_minutes}'{dec_seconds:.2f}\""

    return ra_formatted, dec_formatted


# Function to calculate TLE separation
def calculate_tle_separation():
    try:
        # Get TLE inputs from the text box
        tle_text = tle_input_box.get("1.0", tk.END)
        tle1_line1, tle1_line2, tle2_line1, tle2_line2 = parse_tle_input(tle_text)

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

        # Calculate Euclidean separation distance
        separation = sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)

        # Display result
        result_label.config(text=f"Separation Distance: {separation:.2f} km")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error details: {e}")


# Function to convert TLE to RA/Dec
def convert_tle_to_ra_dec():
    try:
        # Get TLE inputs from the text box
        tle_text = tle_input_box.get("1.0", tk.END)
        tle1_line1, tle1_line2, tle2_line1, tle2_line2 = parse_tle_input(tle_text)

        # Load Skyfield resources
        load = Loader('./skyfield-data')
        ts = load.timescale()

        # Define satellite
        satellite = EarthSatellite(tle1_line1, tle1_line2, "Satellite", ts)

        # Get observation time
        julian_day = float(julian_day_entry.get())
        fractional_day = float(fractional_day_entry.get())
        time = ts.tt_jd(julian_day + fractional_day)

        # Compute RA/Dec
        geocentric = satellite.at(time)
        ra, dec, _ = geocentric.radec()

        # Convert RA/Dec to Stellarium's format
        ra_formatted, dec_formatted = format_ra_dec(ra.hours, dec.degrees)

        # Display RA/Dec
        ra_dec_label.config(text=f"TLE RA: {ra_formatted}, Dec: {dec_formatted}")
        print(f"TLE-derived RA: {ra_formatted}, Dec: {dec_formatted}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        print(f"Error details: {e}")


# GUI Setup
root = tk.Tk()
root.title("Satellite Separation Calculator")

# Tabs
tabs = ttk.Notebook(root)

# TLE Tab
tle_tab = ttk.Frame(tabs)
tabs.add(tle_tab, text="TLE Data")
tabs.pack(expand=1, fill="both")

# TLE Inputs
tk.Label(tle_tab, text="Paste TLE Data:").grid(row=0, column=0, sticky="nw", padx=5, pady=5)
tle_input_box = tk.Text(tle_tab, width=80, height=10)
tle_input_box.grid(row=0, column=1, padx=5, pady=5)

tk.Label(tle_tab, text="Julian Day:").grid(row=1, column=0, sticky="e")
julian_day_entry = tk.Entry(tle_tab, width=20)
julian_day_entry.grid(row=1, column=1)

tk.Label(tle_tab, text="Fractional Day:").grid(row=2, column=0, sticky="e")
fractional_day_entry = tk.Entry(tle_tab, width=20)
fractional_day_entry.grid(row=2, column=1)

# TLE Calculation Button
tle_calculate_button = tk.Button(
    tle_tab, text="Calculate Separation", command=calculate_tle_separation
)
tle_calculate_button.grid(row=3, column=0, columnspan=2, pady=10)

# TLE to RA/Dec Button
tle_ra_dec_button = tk.Button(
    tle_tab, text="Convert TLE to RA/Dec", command=convert_tle_to_ra_dec
)
tle_ra_dec_button.grid(row=4, column=0, columnspan=2, pady=10)

# Result Display for RA/Dec
ra_dec_label = tk.Label(tle_tab, text="TLE RA/Dec: --", font=("Arial", 12))
ra_dec_label.grid(row=5, column=0, columnspan=2, pady=10)

# Result Display for Separation
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=10)

# Start the GUI
root.mainloop()
