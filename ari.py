import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def fsigmoid(x, a, b):
    return 1.0 / (1.0 + np.exp(-a*(x-b)))

#popt, pcov = curve_fit(fsigmoid, xdata, ydata, method='dogbox', bounds=([0., 600.],[0.01, 1200.]))


mydata =  pd.read_csv('C:/Users/andwi/Downloads/covid19cases_test (2).csv')


mydata = mydata[mydata['area'] == 'Los Angeles']

LA = mydata.reset_index(drop = True).iloc[250:425]

cum_cases = LA['cumulative_cases'].to_numpy()
cum_cases = (cum_cases-cum_cases.min())/(cum_cases.max()-cum_cases.min())
popt, pcov = curve_fit(fsigmoid, np.linspace(-3,3,len(cum_cases)), cum_cases)
plt.plot(np.linspace(-3,3,len(cum_cases)),cum_cases)
print(popt)
sample_line = fsigmoid(np.linspace(-3,3,len(cum_cases)),popt[0],popt[1])
plt.scatter(np.linspace(-3,3,len(cum_cases)),cum_cases,s = .1)
plt.plot(np.linspace(-3,3,len(cum_cases)),sample_line)
plt.xlabel("Time Elapsed")
plt.ylabel("Total Infections")
plt.show()

