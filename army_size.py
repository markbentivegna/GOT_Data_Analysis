import csv 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

battles_filename = "game-of-thrones/battles.csv"

fields = [] 
rows = [] 

with open(battles_filename, 'r') as csvfile: 
	csvreader = csv.reader(csvfile) 
	fields = csvreader.__next__()

	for row in csvreader: 
		rows.append(row) 
	print("Total no. of rows: %d"%(csvreader.line_num)) 

print('Field names are:' + ', '.join(field for field in fields)) 

battle_entry = []

for row in rows:
	col_count = 0
	current_attacker_size = 0
	current_defender_size = 0
	for field in row:

		if fields[col_count] == "attacker_size":
			current_attacker_size = field
		if fields[col_count] == "defender_size":
			current_defender_size = field
		if fields[col_count] == "attacker_outcome":
			if field == "win":
				current_outcome = 1
			elif field == "loss":
				current_outcome = 0
		col_count += 1
	if current_outcome and current_attacker_size is not "" and current_defender_size is not "":
		if current_outcome == 1:
			battle_entry.append([int(current_attacker_size), 1])
			battle_entry.append([int(current_defender_size), 0])
		elif current_outcome == 0:
			battle_entry.append([int(current_attacker_size), 0])
			battle_entry.append([int(current_defender_size), 1])

army_size = []
outcome = []
for battle in battle_entry:
	army_size.append(battle[0])
	outcome.append(battle[1])
#Computing the correlation coefficient between army size and battle outcome
r = np.corrcoef(army_size, outcome)
print(r)
#Outputting scatter plot to visualize the relationship between army size and battle outcome
fig, ax = plt.subplots()
scatter = ax.scatter(army_size, outcome)
plt.xlabel("Army Size")
plt.ylabel("Battle Outcome (1: win, 0: defeat)")
plt.show()

#Building training and test sets of the dataset
x = np.array(army_size, ndmin=1)
x = x.reshape(-1, 1).astype(np.float64)
features_train, features_test, labels_train, labels_test = train_test_split(x, outcome, test_size=0.3)
#Using Gaussian NB classifier to try to predict outcomes based solely on army size
NBclf = GaussianNB()
NBclf.fit(features_train, labels_train)
NBpred = NBclf.predict(features_test)
print(accuracy_score(labels_test, NBpred))
#Using Decision Tree classifier to try to predict battle outcomes based solely on army size
DTclf = DecisionTreeClassifier(random_state=5)
DTclf.fit(features_train, labels_train)
DTpred = DTclf.predict(features_test)
print(accuracy_score(labels_test, DTpred))
