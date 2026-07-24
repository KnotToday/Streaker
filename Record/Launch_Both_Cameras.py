import subprocess, os, time, sys

here = os.path.dirname(os.path.abspath(__file__))
exe = os.path.join(here, "dist", "STREAKERrec.exe")
cam2_cfg = os.path.join(here, "stream_capture_config_cam2.json")

subprocess.Popen([exe])
time.sleep(2)
env = os.environ.copy()
env["STREAKER_CONFIG"] = cam2_cfg
subprocess.Popen([exe], env=env)
