"""
Author: Jangwon Park
Date: June 18, 2024
"""
#%% Libraries
# standard libraries
import numpy as np
import pandas as pd
import string
import pickle
np.random.seed(3)

# visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set(style="whitegrid")

# custom python files
from optimization_functions import optimizeCB, runBM
import Erlang_R
import SimRNG

#%% Erlang-R queue experiment. Section 6 of the paper.
# Yom-Tov and Maandelbaum: The Erlang-R Queue (2014), MSOM.

# Import data: cumulative arrivals and depatures
cum_arrivals = pd.read_csv('MCE_cumulative_arrivals.csv')
cum_departures = pd.read_csv('MCE_cumulative_departures.csv')

# Plot cumulative arrivals and departures (Figure 10 (a) of Yom-Tov and Mandelbaum, 2014)
cum_arrivals['Time'] = pd.to_datetime(cum_arrivals['Time'])
cum_departures['Time'] = pd.to_datetime(cum_departures['Time'])
cum_departures['Time'] = pd.to_datetime(cum_departures['Time'])

plt.close('all')
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(cum_arrivals['Time'][1:], cum_arrivals['Cumulative Arrivals'][1:], 
           marker='D', markersize=3.5, label='Cumulative arrivals', alpha=1, color='k')
ax.plot(cum_departures['Time'][1:], cum_departures['Cumulative Departures'][1:], 
           marker='s', markersize=3.5, label='Cumulative departures', alpha=0.5)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_xlabel('Time [hour:min]')
ax.set_ylabel('Total number of patients')
ax.set_ylim(0, 60)
ax.grid(axis='y', linestyle=':')
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()

# Plot the arrival rate function 
ts = np.linspace(0,120,3000)
arrival_rate_fn = lambda t: 0.773*(0<=t<22) + 0.884*(44<=t<69) + 0.5*(102<=t<117)

change_points = np.array([0, 22, 44, 69, 102, 117])  # x-coordinates of changes
values = np.array([0.773, 0, 0.884, 0, 0.5, 0])  # corresponding y-values at each segment

# Create x values for plotting
x = []
y = []
for i in range(len(change_points)-1):
    x.extend([change_points[i], change_points[i+1]])
    y.extend([values[i], values[i]])

# Add the last value to cover the last segment
x.append(change_points[-1])
y.append(values[-1])

# Plotting the piecewise constant function
plt.figure(figsize=(7,5))

# Adding circular markers
plt.scatter(change_points[1:], values[:-1], color='k', s=25, facecolors='none', zorder=5, edgecolors='black') # empty
plt.scatter(change_points[1:], values[1:], facecolors='k', edgecolors='black', s=25, zorder=5) # filled

# Add horizontal lines
for i in range(len(values)-1):
    plt.axhline(values[i], color='k', xmin=change_points[i]/120, xmax=change_points[i+1]/120)
plt.axhline(0, color='k', xmin=change_points[-1]/120, xmax=1)

# Add vertical dotted lines at discontinuities
pad = 0.05
tmp = [(pad,0.773), (pad,0.884), (pad,0.884), (pad,0.5), (pad,0.5)]
for i in range(len(change_points)-1):
    plt.axvline(x=change_points[i+1], ymin=tmp[i][0], ymax=tmp[i][1], color='k', linestyle=':', linewidth=1)

# Labels and title
plt.xlabel(r'$t$')
plt.ylabel(r'$\lambda(t)$')
plt.grid(alpha=0.3)
plt.xlim(0, 120)
plt.ylim(-pad, 1)
plt.tight_layout()
plt.show()

#%% Erlang-R simulation

params = {'N': 4, # number of servers
          'arrival_rate_fn': arrival_rate_fn,
          'max_rate': 0.884, # max. arrival rate
          'mu': 11.06/60, # service rate
          'delta': 2.44/60, # delay rate (content state)
          'p': 0.662, # prob. of re-entrance
          } 

SimRNG.ZRNG = SimRNG.InitializeRNSeed() # reset

reps = 300 # number of simulation replications
Q = []
t_list = []
num_events = []
for r in range(reps):
    y, x, t = Erlang_R.simulate(params)
    Q.append(x+y)
    t_list.append(t)
    num_events.append(len(t))

# Create sample paths with equidistant time steps
H = 30 # number of time steps
print('H:', H)
ts = np.linspace(0, 120, H)

samples = []

for r in range(reps):
    tmp = []
    for t in ts:
        idx = np.searchsorted(t_list[r], t, side='right') - 1
        tmp.append(Q[r][idx])
    samples.append(np.array(tmp))
samples = np.array(samples)

#%% Construct CBs
plt.close('all')
K = 3 # number of folds in cross validation
alpha = 0.05 # significance level

# Construct robust CB
solve_again = False # if False, assumes pre-solved confidence bands exist.

if solve_again:
    Gamma = runBM(samples, K, alpha)
    print("Gamma:", Gamma)
    UBr, LBr, _ = optimizeCB(samples, alpha=alpha, Gamma=Gamma, gap=0.05)

    # Save CB
    with open("UBr_" + str(len(samples)) + "_" + str(int(100*alpha)), "wb") as fp: pickle.dump(UBr, fp)
    with open("LBr_" + str(len(samples)) + "_" + str(int(100*alpha)), "wb") as fp: pickle.dump(LBr, fp)
else:
    # Load CB
    with open("UBr_" + str(len(samples)) + "_" + str(int(100*alpha)), "rb") as fp: UBr = pickle.load(fp)
    with open("LBr_" + str(len(samples)) + "_" + str(int(100*alpha)), "rb") as fp: LBr = pickle.load(fp)

#%% Draw CB
# Get actual dates from actual data
all_dates = pd.DataFrame({'Time': cum_arrivals['Time'][1:], 'Increment': [1]*(cum_arrivals.shape[0]-1)})
all_dates = pd.concat([all_dates, pd.DataFrame({'Time': cum_departures['Time'][1:], 'Increment': [-1]*(cum_departures.shape[0]-1)})])
all_dates = all_dates.sort_values(by='Time')

def convertToTimes(t_list, start=all_dates['Time'].iloc[0], end=all_dates['Time'].iloc[-1]):
    """Convert time indices to actual dates."""
    return [start + (end-start)*(t/120)  for t in t_list]

plt.close('all')
fig, ax = plt.subplots(figsize=(7,5))
for r in range(reps):
    ax.plot(convertToTimes(t_list[r]), Q[r], drawstyle='steps-post', linewidth=0.5, alpha=0.5)

# Plot actual path    
ax.plot(all_dates['Time'], np.cumsum(all_dates['Increment']), color='k', linewidth=2,
        drawstyle='steps-post', label=r'$Q(t)$')

# Plot CB (densely dotted=(0,(1,1)))
times = convertToTimes(ts)
ax.plot(times, UBr, linewidth=2, color="blue", linestyle=(0,(1,1)), label=r"95% confidence band ($\alpha=0.05$)")
ax.plot(times, LBr, linewidth=2, color="blue", linestyle=(0,(1,1)),)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.set_ylabel('Total number of patients')
ax.set_xlabel('Time [hour:min]')
plt.legend()
plt.tight_layout()
plt.show()
