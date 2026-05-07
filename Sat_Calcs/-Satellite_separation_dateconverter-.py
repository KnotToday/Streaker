import tkinter as tk

from tkinter import messagebox

from skyfield.api import Loader, EarthSatellite

from math import sqrt

from datetime import datetime, timedelta


# Function to calculate Julian Day and Fractional Day

def utc_to_julian():

    try:

        # Get UTC inputs

        year = int(year_entry.get())

        month = int(month_entry.get())

        day = int(day_entry.get())

        hour = int(hour_entry.get())

        minute = int(minute_entry.get())

        second = int(second_entry.get())


        # Convert to Julian Day and Fractional Day

        dt = datetime(year, month, day, hour, minute, second)

        julian_epoch = datetime(2000, 1, 1, 12)  # Julian Day 2451545.0

        days_since_epoch = (dt - julian_epoch).days

        fractional_day = (dt - julian_epoch).seconds / 86400

        julian_day = 2451545 + days_since_epoch + fractional_day


        # Display results

        result_label.config(text=f"Julian Day: {int(julian_day)}\nFractional Day: {julian_day - int(julian_day):.6f}")

    except Exception as e:

        messagebox.showerror("Error", f"An error occurred: {e}")


# Function to calculate satellite separation (unchanged)

def calculate_separation():

    try:

        # Get TLE data from input fields

        tle1_line1 = tle1_line1_entry.get().strip()

        tle1_line2 = tle1_line2_entry.get().strip()

        tle2_line1 = tle2_line1_entry.get().strip()

        tle2_line2 = tle2_line2_entry.get().strip()


        # Get Julian and Fractional Day inputs

        julian_day = float(julian_day_entry.get())

        fractional_day = float(fractional_day_entry.get())


        # Load Skyfield resources

        load = Loader('./skyfield-data')

        ts = load.timescale()


        # Define satellites

        satellite1 = EarthSatellite(tle1_line1, tle1_line2, "Satellite 1", ts)

        satellite2 = EarthSatellite(tle2_line1, tle2_line2, "Satellite 2", ts)


        # Define observation time

        time = ts.tt_jd(julian_day + fractional_day)


        # Compute geocentric positions

        pos1 = satellite1.at(time).position.km

        pos2 = satellite2.at(time).position.km


        # Calculate Euclidean distance

        separation = sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)


        # Display result

        result_label.config(text=f"Separation Distance: {separation:.2f} km")

    except Exception as e:

        messagebox.showerror("Error", f"An error occurred: {e}")


# GUI Setup

root = tk.Tk()

root.title("Satellite Separation Calculator")


# TLE Inputs for Satellite 1

tk.Label(root, text="Satellite 1 TLE Line 1:").grid(row=0, column=0, padx=5, pady=5, sticky="e")

tle1_line1_entry = tk.Entry(root, width=60)

tle1_line1_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Satellite 1 TLE Line 2:").grid(row=1, column=0, padx=5, pady=5, sticky="e")

tle1_line2_entry = tk.Entry(root, width=60)

tle1_line2_entry.grid(row=1, column=1, padx=5, pady=5)


# TLE Inputs for Satellite 2

tk.Label(root, text="Satellite 2 TLE Line 1:").grid(row=2, column=0, padx=5, pady=5, sticky="e")

tle2_line1_entry = tk.Entry(root, width=60)

tle2_line1_entry.grid(row=2, column=1, padx=5, pady=5)

tk.Label(root, text="Satellite 2 TLE Line 2:").grid(row=3, column=0, padx=5, pady=5, sticky="e")

tle2_line2_entry = tk.Entry(root, width=60)

tle2_line2_entry.grid(row=3, column=1, padx=5, pady=5)


# UTC Inputs for Julian Day Conversion

tk.Label(root, text="UTC Year:").grid(row=4, column=0, padx=5, pady=5, sticky="e")

year_entry = tk.Entry(root, width=10)

year_entry.grid(row=4, column=1, padx=5, pady=5, sticky="w")

year_entry.insert(0, "2024")


tk.Label(root, text="Month:").grid(row=5, column=0, padx=5, pady=5, sticky="e")

month_entry = tk.Entry(root, width=10)

month_entry.grid(row=5, column=1, padx=5, pady=5, sticky="w")

month_entry.insert(0, "12")


tk.Label(root, text="Day:").grid(row=6, column=0, padx=5, pady=5, sticky="e")

day_entry = tk.Entry(root, width=10)

day_entry.grid(row=6, column=1, padx=5, pady=5, sticky="w")

day_entry.insert(0, "14")


tk.Label(root, text="Hour:").grid(row=7, column=0, padx=5, pady=5, sticky="e")

hour_entry = tk.Entry(root, width=10)

hour_entry.grid(row=7, column=1, padx=5, pady=5, sticky="w")

hour_entry.insert(0, "6")


tk.Label(root, text="Minute:").grid(row=8, column=0, padx=5, pady=5, sticky="e")

minute_entry = tk.Entry(root, width=10)

minute_entry.grid(row=8, column=1, padx=5, pady=5, sticky="w")

minute_entry.insert(0, "22")


tk.Label(root, text="Second:").grid(row=9, column=0, padx=5, pady=5, sticky="e")

second_entry = tk.Entry(root, width=10)

second_entry.grid(row=9, column=1, padx=5, pady=5, sticky="w")

second_entry.insert(0, "0")


# Calculate Julian Day Button

convert_button = tk.Button(root, text="Convert to Julian Day", command=utc_to_julian)

convert_button.grid(row=10, column=0, columnspan=2, pady=10)


# Julian and Fractional Day Inputs

tk.Label(root, text="Julian Day:").grid(row=11, column=0, padx=5, pady=5, sticky="e")

julian_day_entry = tk.Entry(root, width=20)

julian_day_entry.grid(row=11, column=1, padx=5, pady=5, sticky="w")


tk.Label(root, text="Fractional Day:").grid(row=12, column=0, padx=5, pady=5, sticky="e")

fractional_day_entry = tk.Entry(root, width=20)

fractional_day_entry.grid(row=12, column=1, padx=5, pady=5, sticky="w")


# Calculate Separation Button

calculate_button = tk.Button(root, text="Calculate Separation", command=calculate_separation)

calculate_button.grid(row=13, column=0, columnspan=2, pady=10)


# Result Display

result_label = tk.Label(root, text="", font=("Arial", 14))

result_label.grid(row=14, column=0, columnspan=2, pady=10)


# Start the Tkinter event loop

root.mainloop()