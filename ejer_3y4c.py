import subprocess
import numpy as np
import matplotlib.pyplot as plt

r = np.array([1/64, 1/16, 1/4, 1, 4, 16, 64])
mean_pos_err_ekf = []
anees_ekf = []
mean_pos_err_pf = []
anees_pf = []
for i in range(len(r)):
    command = "python3 localization.py ekf --filter-factor %f --seed 0" % (r[i]) 
    output = (subprocess.check_output(command, shell=True).decode("utf-8")) #.split("\n")
    print(output)
    idx = output.find("Mean position error: ")
    ME = float(output[idx+21:idx+21+9])
    mean_pos_err_ekf.append(ME)
    idx = output.find("ANEES: ")
    ANEES = float(output[idx+7:idx+7+9])
    anees_ekf.append(ANEES)

    command = "python3 localization.py pf --filter-factor %f --seed 0" % (r[i])
    output = (subprocess.check_output(command, shell=True).decode("utf-8")) #.split("\n")
    print(output)
    idx = output.find("Mean position error: ")
    ME = float(output[idx+21:idx+21+9])
    mean_pos_err_pf.append(ME)
    idx = output.find("ANEES: ")
    ANEES = float(output[idx+7:idx+7+9])
    anees_pf.append(ANEES)

default_x_ticks = range(len(r))
fig3, ax3 = plt.subplots(2, sharex='col')
ax3[0].plot(default_x_ticks, mean_pos_err_ekf, 'b')
ax3[1].plot(default_x_ticks, anees_ekf, 'r')
fig3.suptitle('EKF', fontsize=16)
ax3[0].set_ylabel('Mean Pos Error')
ax3[0].grid()
ax3[1].set_ylabel('ANEES')
ax3[1].grid()
ax3[1].set_xlabel('r')
plt.xticks(default_x_ticks, ['1/64', '1/16', '1/4', '1', '4', '16', '64'])
plt.xticks(default_x_ticks, r)
fig4, ax4 = plt.subplots(2, sharex='col')
ax4[0].plot(default_x_ticks, mean_pos_err_pf, 'b')
ax4[1].plot(default_x_ticks, anees_pf, 'r')
fig4.suptitle('PF', fontsize=16)
ax4[0].set_ylabel('Mean Pos Error')
ax4[0].grid()
ax4[1].set_ylabel('ANEES')
ax4[1].grid()
ax4[1].set_xlabel('r')
plt.xticks(default_x_ticks, ['1/64', '1/16', '1/4', '1', '4', '16', '64'])
plt.xticks(default_x_ticks, r)
plt.show(block=True)

