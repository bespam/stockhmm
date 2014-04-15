from __future__ import print_function

import datetime
import numpy as np
import pylab as pl
from matplotlib.finance import quotes_historical_yahoo
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from sklearn.hmm import GaussianHMM
import pdb
import itertools



###############################################################################
# Downloading the data
date1 = datetime.date(2008, 1, 1)  # start date
date2 = datetime.date(2014, 1, 14)  # end date
# get quotes from yahoo finance
quotes1 = quotes_historical_yahoo("JPM", date1, date2)
quotes2 = quotes_historical_yahoo("WFC", date1, date2)
print(len(quotes1))
if len(quotes1) == 0 or len(quotes1) != len(quotes2):
    raise SystemExit

dates = np.array([q[0] for q in quotes1], dtype=int)
dates = dates[1:]

# unpack quotes
close_v1 = np.array([q[2] for q in quotes1])
volume1 = np.array([q[5] for q in quotes1])[1:]
close_v2 = np.array([q[2] for q in quotes2])
volume2 = np.array([q[5] for q in quotes2])[1:]

# take diff of close value
# this makes len(diff) = len(close_t) - 1
# therefore, others quantity also need to be shifted
diff1 = close_v1[1:] - close_v1[:-1]
close_v1 = close_v1[1:]
diff2 = close_v2[1:] - close_v2[:-1]
close_v2 = close_v2[1:]

# pack diff and volume for training
X1 = np.column_stack([diff1, volume1])
X2 = np.column_stack([diff2, volume2])
###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...", end='')
n_components = 5
# make an HMM instance and execute fit
model1 = GaussianHMM(n_components, covariance_type="diag", n_iter=1000)
model2 = GaussianHMM(n_components, covariance_type="diag", n_iter=1000)

model1.fit([X1])
model2.fit([X2])

# predict the optimal sequence of internal hidden state
hidden_states1 = model1.predict(X1)
hidden_states2 = model2.predict(X2)

print("done\n")

# calculate similarity measure
states1 = range(n_components)
states2 = list(itertools.permutations(states1))
print(states1)
print(len(states2))
sims = []
for i in range(len(states2)):
    sim = 0
    for j in range(len(hidden_states1)):
        sim += hidden_states1[j] == states2[i][hidden_states2[j]]
        #pdb.set_trace()
    sims.append(float(sim)/len(hidden_states1))

similarity = max(sims)    
print(["similarity: ", similarity])
m_ind = sims.index(similarity)
st = states2[m_ind]

###############################################################################
# print trained parameters and plot
print("Transition matrix")
print(model1.transmat_)
print()

print("means and vars of each hidden state")
for i in range(n_components):
    print("%dth hidden state" % i)
    print("mean = ", model1.means_[i])
    print("var = ", np.diag(model1.covars_[i]))
    print()

years = YearLocator()   # every year
months = MonthLocator()  # every month
yearsFmt = DateFormatter('%Y')
fig = pl.figure()
ax = fig.add_subplot(111)
colors = ['r','b','g','m','k']

for i in range(n_components):
    # use fancy indexing to plot data in each state
    idx = (hidden_states1 == i)
    ax.plot_date(dates[idx], close_v1[idx], 'o', label="%dth hidden state" % i, color = colors[i])
ax.legend(loc=2)

#used_colors = ax._get_lines.color_cycle
#pdb.set_trace()

for i in range(n_components):
    # use fancy indexing to plot data in each state
    idx = (hidden_states2 == st[i])
    ax.plot_date(dates[idx], close_v2[idx], 'o', label="%dth hidden state" % i, color = colors[i])


# format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)
ax.xaxis.set_minor_locator(months)
ax.autoscale_view()

# format the coords message box
ax.fmt_xdata = DateFormatter('%Y-%m-%d')
ax.fmt_ydata = lambda x: '$%1.2f' % x
ax.grid(True)

fig.autofmt_xdate()
pl.show()



#pl.plot(range(len(sims)),sims)
#pl.show()

