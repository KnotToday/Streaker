import tkinter as tk

from tkinter import messagebox

from sgp4.api import Satrec

from math import sqrt

import sys

# Print the path to the Python executable 
print("Python executable being used:", sys.executable)

# Function to calculate separation distance

def calculate_distance():

    try:

        # Get input data from text fields

        tle_1_line1 = tle1_line1_entry.get("1.0", tk.END).strip()

        tle_1_line2 = tle1_line2_entry.get("1.0", tk.END).strip()

        tle_2_line1 = tle2_line1_entry.get("1.0", tk.END).strip()

        tle_2_line2 = tle2_line2_entry.get("1.0", tk.END).strip()

        

        # Parse TLE data

        sat1 = Satrec.twoline2rv(tle_1_line1, tle_1_line2)

        sat2 = Satrec.twoline2rv(tle_2_line1, tle_2_line2)


        # Define epoch time (Julian date and fractional day)

        jd = float(epoch_jd_entry.get())

        fr = float(epoch_fr_entry.get())


        # Get positions

        e1, r1, _ = sat1.sgp4(jd, fr)

        e2, r2, _ = sat2.sgp4(jd, fr)


        # Check for errors

        if e1 != 0 or e2 != 0:

            messagebox.showerror("Error", f"Propagation error: Sat1 = {e1}, Sat2 = {e2}")

            return


        # Calculate separation distance

        distance = sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2 + (r1[2] - r2[2])**2)


        # Display result

        result_label.config(text=f"Distance: {distance:.2f} km")


    except Exception as e:

        messagebox.showerror("Error", f"An error occurred: {e}")


# GUI setup

root = tk.Tk()

root.title("Satellite Separation Calculator")


# TLE Inputs

tk.Label(root, text="Satellite 1 TLE").grid(row=0, column=0, columnspan=2)

tk.Label(root, text="Line 1:").grid(row=1, column=0, sticky="e")

tle1_line1_entry = tk.Text(root, width=70, height=1)

tle1_line1_entry.grid(row=1, column=1)


tk.Label(root, text="Line 2:").grid(row=2, column=0, sticky="e")

tle1_line2_entry = tk.Text(root, width=70, height=1)

tle1_line2_entry.grid(row=2, column=1)


tk.Label(root, text="Satellite 2 TLE").grid(row=3, column=0, columnspan=2)

tk.Label(root, text="Line 1:").grid(row=4, column=0, sticky="e")

tle2_line1_entry = tk.Text(root, width=70, height=1)

tle2_line1_entry.grid(row=4, column=1)


tk.Label(root, text="Line 2:").grid(row=5, column=0, sticky="e")

tle2_line2_entry = tk.Text(root, width=70, height=1)

tle2_line2_entry.grid(row=5, column=1)


# Epoch Inputs

tk.Label(root, text="Epoch Time").grid(row=6, column=0, columnspan=2)

tk.Label(root, text="Julian Day:").grid(row=7, column=0, sticky="e")

epoch_jd_entry = tk.Entry(root, width=20)

epoch_jd_entry.grid(row=7, column=1, sticky="w")

epoch_jd_entry.insert(0, "24349")  # Default value


tk.Label(root, text="Fractional Day:").grid(row=8, column=0, sticky="e")

epoch_fr_entry = tk.Entry(root, width=20)

epoch_fr_entry.grid(row=8, column=1, sticky="w")

epoch_fr_entry.insert(0, "0.79407039")  # Default value


# Calculate Button

calculate_button = tk.Button(root, text="Calculate Distance", command=calculate_distance)

calculate_button.grid(row=9, column=0, columnspan=2, pady=10)


# Result Display

result_label = tk.Label(root, text="", font=("Arial", 14))

result_label.grid(row=10, column=0, columnspan=2, pady=10)


# Run the GUI

root.mainloop()