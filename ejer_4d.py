import subprocess
import numpy as np
import matplotlib.pyplot as plt

r = np.array([1/64, 1/16, 1/4, 1, 4, 16, 64]) 
part = np.array([20, 50, 500])
mean_pos_err_pf = np.zeros((len(part), len(r)))
anees_pf = np.zeros((len(part), len(r)))

for j in range(len(part)):
    for i in range(len(r)):
        command = "python3 localization.py pf --filter-factor %f --num-particles %d --seed 0" % (r[i], part[j])
        output = (subprocess.check_output(command, shell=True).decode("utf-8"))
        # print(output)
        idx = output.find("Mean position error: ")
        ME = float(output[idx+21:idx+21+9])
        mean_pos_err_pf[j][i] = ME
        idx = output.find("ANEES: ")
        ANEES = float(output[idx+7:idx+7+9])
        anees_pf[j][i] = ANEES

default_x_ticks = range(len(r))
fig, ax = plt.subplots(nrows=3, ncols=2)
idx = 0
for row in ax:
    titles = ["Mean Error - %d Particles" % (part[idx]), "ANEES %d Particles" % (part[idx])]
    row[0].plot(default_x_ticks, mean_pos_err_pf[idx], 'b')
    row[0].set_title(titles[0])
    row[0].grid()
    row[0].set_xticks(default_x_ticks, ['1/64', '1/16', '1/4','1', '4', '16', '64'])
    row[1].plot(default_x_ticks, anees_pf[idx], 'r')
    row[1].set_title(titles[1])
    row[1].grid()
    row[1].set_xticks(default_x_ticks, ['1/64', '1/16', '1/4','1', '4', '16', '64'])
    idx+=1

plt.show(block=True)

