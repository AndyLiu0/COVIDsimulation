import numpy as np
from numpy import savetxt
#from numpy import loadtxt
import matplotlib.pyplot
import matplotlib.ticker as mtick

tests = 5 #5
trials = 5
stop = 5000
x = np.arange(stop)
num_characters = 1000
#fig, axs = matplotlib.pyplot.subplots(ncols=3, layout='constrained', figsize=[13.0, 5.0])
fig, axs = matplotlib.pyplot.subplots(ncols=2, layout='constrained', figsize=[13.0, 5.0])

axs[0].set(xlim=(0, stop), ylim=(0, num_characters))
axs[0].set_xlabel("Time")
axs[0].set_ylabel("# of People Infected")

x_2 = [0.001, 0.0005, 0.0003, 0.0002, 0.0001]

axs[1].set(xlim=(np.amin(x_2) - (np.amax(x_2) - np.amin(x_2)) / 10, np.amax(x_2) + (np.amax(x_2) - np.amin(x_2)) / 10), ylim=(0, num_characters))
#axs[2].set(xlim=(np.amin(x_2) - (np.amax(x_2) - np.amin(x_2)) / 10, np.amax(x_2) + (np.amax(x_2) - np.amin(x_2)) / 10), ylim=(0, 5000))

print(len(x_2))
y_2 = []
yerr_2 = []

y_3 = []
yerr_3 = []

axs[1].set_xticks(x_2)
axs[1].set_xlabel("Migration Chance")
axs[1].set_ylabel("Total # of People Infected")
axs[1].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

# axs[2].set_xticks(x_2)
# axs[2].set_xlabel("Migration Chance")
# axs[2].set_ylabel("Time until 90% Threshold")
# axs[2].xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

axs[0].set_title('A', loc='left')
axs[1].set_title('B', loc='left')
# axs[2].set_title('C', loc='left')


colors_list = ['r', 'y', 'g', 'b', 'm']
sum_data = np.load('MigrationChance.npy', allow_pickle=False)
print(sum_data.shape)
#print(type(sum_data))
#print(sum_data)
#print(sum_data.shape)
avg_infected = np.zeros(5000)
for a in range(tests):
    trials_list = []
    max_list = []
    threshold_list = []
    for b in range(trials):
        infected_arr, incubation_arr, immune_arr, alive_arr, dead_arr, r_arr, immunity_arr = sum_data[a + 1].T[b]
        total_infected = np.sum((infected_arr, incubation_arr, immune_arr, dead_arr), axis=0)
        trials_list.append(total_infected)
        print(b)
        print(np.amax(total_infected))
        max_list.append(np.amax(total_infected))
        #axs[0].plot(x, total_infected, (255, 0, 0))
        threshold_list.append(np.argmax(total_infected > np.amax(total_infected) * 0.9))

    avg_infected = np.sum(trials_list, axis=0)/trials
    print("test" + str(a))
    print(np.amax(avg_infected))
    print(avg_infected.shape)
    axs[0].plot(x, avg_infected, colors_list[a])

    y_2.append(sum(max_list)/trials)
    yerr_2.append(np.std(max_list))

    y_3.append(sum(threshold_list)/trials)
    yerr_3.append(np.std(threshold_list))
    #print(a + 1)

#x_2 = int(x_2)
#print(len(x_2))
#print(len(y_2))
axs[1].scatter(x_2, y_2, c='k')
axs[1].errorbar(x_2, y_2, yerr=yerr_2, capsize=3)
#
# axs[2].scatter(x_2, y_3, c='k')
# axs[2].errorbar(x_2, y_3, yerr=yerr_3, capsize=3)

labels = []
for i in range(tests):
    labels.append(str(x_2[i]*100) + "%")
    #labels.append(x_2[i])
axs[0].legend(labels, loc='best')
print(y_2)
matplotlib.pyplot.show()