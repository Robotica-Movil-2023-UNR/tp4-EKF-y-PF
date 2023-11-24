import subprocess
import numpy as np
import matplotlib.pyplot as plt

command = "python3 localization.py ekf --plot"
subprocess.check_output(command, shell=True).decode("utf-8")
command = "python3 localization.py pf --plot"
subprocess.check_output(command, shell=True).decode("utf-8")