# Troubleshooting ideas for missing progress bar in a Tkinter app launched from another script

# 1. Ensure the theme supports ttk.Progressbar
#    Try switching to a classic theme for compatibility testing
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
style = ttk.Style(root)
print("Available themes:", style.theme_names())
style.theme_use('vista')  # Try 'classic' or 'default' if 'vista' causes problems

# 2. Confirm parent window is being displayed properly
print("Master widget visible:", root.winfo_viewable())

# 3. Add debugging prints before and after progress updates
#    to ensure control is reaching those lines
print("Setting progress value")
progress = ttk.Progressbar(root, length=200)
progress.pack()
progress["maximum"] = 10
progress["value"] = 5

# 4. Force GUI update to make sure UI reflects progress
root.update_idletasks()

# 5. Verify if script is being run in the main thread
import threading
print("Is main thread:", threading.current_thread() is threading.main_thread())

# Run a mainloop to see the effect
root.mainloop()
