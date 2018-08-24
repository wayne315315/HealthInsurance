import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

filename = 'insurance.csv'
foldername = 'asset'


with open(filename, mode='rt', newline='') as f:
	reader = csv.reader(f, delimiter=',')
	scheme = next(reader)
	data = np.array([row for row in reader])

# Data scheme
# ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']
# Categorical variables : sex, smoker (?), region
### sex : male, female
### smoker : yes, no
### region : northeast, southwest, southeast, northwest
# Numerical variables : age, bmi, children #
# Target : charges (Numerical)

# sex
def raw2bar(feature, column):
	counts = {}
	for attr in column:
		try:
			counts[attr] += 1
		except KeyError:
			counts[attr] = 1
	labels = list(counts.keys())
	values = [counts[label] for label in labels]
	indices = list(range(len(labels)))

	print('\n*** Feature : ', feature)
	for label, value in zip(labels, values):
		print('- %s : %i' % (label, value))


	if not os.path.exists(foldername):
		os.mkdir(foldername)

	plt.bar(indices, values)
	plt.xlabel(feature, fontsize=8)
	plt.ylabel('count', fontsize=8)
	plt.xticks(indices, labels, fontsize=8)
	plt.title('Individual count for each %s' % feature)

	ax = plt.axes()
	for x, val in enumerate(values):
		ax.text(x, val + 3, str(val), color='blue', fontweight='bold')

	filepath = os.path.join(foldername, feature + '.png')
	plt.savefig(filepath, format='png')
	plt.close()

def raw2box(feature, column):
	column = column.astype(float)
	y_units = {'age':'year', 'bmi':'kg/m^2', 'children':'', 'charges':'dollar'}
	unit = y_units[feature]

	f_mean = np.mean(column)
	f_sd = np.std(column)

	f_median = np.median(column)
	f_q1 = np.percentile(column, 25)
	f_q3 = np.percentile(column, 75)

	f_min =  np.amin(column)
	f_max =  np.amax(column)

	names = ['mean', 'sd', 'median', 'q1', 'q3', 'min', 'max']
	statistics = [f_mean, f_sd, f_median, f_q1, f_q3, f_min, f_max]

	print('\n*** Feature : ', feature)
	for name, stat in zip(names, statistics):
		print('- %s : %.3f' % (name, stat))

	plt.boxplot(column, showmeans=True)
	plt.title(feature)
	plt.ylabel(unit)

	filepath = os.path.join(foldername, feature + '.png')
	plt.savefig(filepath, format='png')
	plt.close()

i = 0
while i < len(scheme):
	feature = scheme[i]
	column = data[:, i]
	categoricals = ['sex', 'smoker', 'region']
	numericals = ['age', 'bmi', 'children', 'charges']
	if feature in categoricals:
		raw2bar(feature, column)
	else:
		raw2box(feature, column)
	i += 1

def shapiro_normalize_test(feature):
	for i, f in enumerate(scheme):
		if f == feature:
			column = data[:, i]
	_, p_val = stats.shapiro(column)
	print('*** %s normality test' % feature)
	print('p value : %.3f' % p_val)
	if p_val < 0.05 :
		print('Normal distribution : Yes')
	else:
		print('Normal distribution : No')

for feature in numericals:
	shapiro_normalize_test(feature)
# Age, bmi, childer number all sucessfully pass normality test. Unfortunatelly, 
# charges is rejected as being normal distributed

# Correlation coefficient 
# - age vs. bmi
reverse_scheme = {f:i for i, f in enumerate(scheme)}
corr = r2_score(data[:, reverse_scheme['age']].astype(float), data[:, reverse_scheme['bmi']].astype(float))
print("*** age vs. bmi")
print("Correlation coefficient : ", corr)

