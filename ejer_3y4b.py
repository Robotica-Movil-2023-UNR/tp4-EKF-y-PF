import subprocess
import numpy as np
import matplotlib.pyplot as plt

r = np.array([1/64, 1/16, 1/4, 1, 4, 16, 64])
mean_pos_err_ekf = []
mean_pos_err_pf = []
for i in range(len(r)):
    command = "python3 localization.py ekf --data-factor %f --filter-factor %f" % (r[i], r[i]) # --seed 0
    output = (subprocess.check_output(command, shell=True).decode("utf-8")) #.split("\n")
    idx = output.find("Mean position error: ")
    ME = float(output[idx+21:idx+21+9])
    mean_pos_err_ekf.append(ME)

    command = "python3 localization.py pf --data-factor %f --filter-factor %f" % (r[i], r[i])  #--seed 0
    output = (subprocess.check_output(command, shell=True).decode("utf-8")) #.split("\n")
    idx = output.find("Mean position error: ")
    ME = float(output[idx+21:idx+21+9])
    mean_pos_err_pf.append(ME)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
fig1.suptitle("Error medio de posición - EKF")
default_x_ticks = range(len(r))
ax1.plot(default_x_ticks, mean_pos_err_ekf, 'b')
plt.xticks(default_x_ticks, ['1/64', '1/16', '1/4', '1', '4', '16', '64'])
ax1.set_ylabel('[mts]')
ax1.set_xlabel('r')
ax1.grid()
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
fig2.suptitle("Error medio de posición - PF")
default_x_ticks = range(len(r))
ax2.plot(default_x_ticks, mean_pos_err_pf, 'b')
plt.xticks(default_x_ticks, ['1/64', '1/16', '1/4', '1', '4', '16', '64'])
ax2.set_ylabel('[mts]')
ax2.set_xlabel('r')
ax2.grid()
plt.show(block=True)

    # ANEES = float(output[6][7:])
