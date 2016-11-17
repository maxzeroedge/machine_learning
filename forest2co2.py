import numpy as np
from svmutil import *
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

class Forest2CO2:
	def __init__(self, train_file):
		with open(train_file, 'r') as f:
			inp = f.readlines()
			inp = [n.strip('\n').split(',') for n in inp]
			labels = inp[0]
			inp = inp[1:]
			out = []
			for i in range(0, len(inp)):
				out.append(inp[i][0])
				inp[i] = [float(a) for a in inp[i][1:9]]
			self.classes = []
			for i in range(0, len(out)):
				if(out[i] in self.classes):
					out[i] = 1 + self.classes.index(out[i])
					continue
				else:
					self.classes.append(out[i])
					out[i] = 1 + self.classes.index(out[i])
			self.train_X = inp
			self.train_Y = out

	def train_data(self):
		# self.model = svm.NuSVC(nu=0.01,kernel='rbf', gamma=0.004, tol=0.00001, probability=True)
		# Error = 0.44
		self.model = svm.SVC(C=0.05, kernel='rbf', gamma=0.125, tol=0.00001, probability=True)
		# c = 10, g=0.004: 0.4338
		# c = 0.1: 0.3877
		# c = 0.01: 0.3810
		# c = 0.05, g = 0.01: 0.37
		# c = 0.05, g = 0.1: 0.09
		# g = 0.12: 0.0865
		# g = 0.125: 0.0865
		# g = 0.13: 0.089
		self.model = OneVsRestClassifier(self.model).fit(self.train_X, self.train_Y)

	def test_data(self, test_file):
		with open(test_file, 'r') as f:
			inp = f.readlines()
			inp = [n.strip('\n').split(',') for n in inp]
			labels = inp[0]
			inp = inp[1:]
			out = []
			for i in range(0, len(inp)):
				out.append(inp[i][0])
				inp[i] = [float(a) for a in inp[i][1:9]]
			self.classes = []
			for i in range(0, len(out)):
				if(out[i] in self.classes):
					out[i] = 1 + self.classes.index(out[i])
					continue
				else:
					self.classes.append(out[i])
					out[i] = 1 + self.classes.index(out[i])
			pred = self.model.predict(inp)
			# pred = svm_predict(out, inp, self.model)
			# return np.sum(out-pred)/len(out)
			print(float(np.sum(np.absolute(pred==out)))/float(len(out)))
			co2vals = []
			for p in pred:
				if(p == 1):
					# Sugi type: https://books.google.co.in/books?id=wKxbmMe_lDsC&pg=PA11&lpg=PA11&dq=carbon+storage+potential+of+sugi+forest&source=bl&ots=OHr9wnQKP4&sig=UXL2VcPQlpyFaQOC8mugBl8w8zU&hl=en&sa=X&ved=0ahUKEwjL6O7Y1MPMAhWIB44KHS2MAtgQ6AEIIDAA#v=onepage&q=carbon%20storage%20potential%20of%20sugi%20forest&f=false
					co2vals.append('4.3')
				elif(p == 2):
					# Hinoki type: https://books.google.co.in/books?id=wKxbmMe_lDsC&pg=PA11&lpg=PA11&dq=carbon+storage+potential+of+sugi+forest&source=bl&ots=OHr9wnQKP4&sig=UXL2VcPQlpyFaQOC8mugBl8w8zU&hl=en&sa=X&ved=0ahUKEwjL6O7Y1MPMAhWIB44KHS2MAtgQ6AEIIDAA#v=onepage&q=carbon%20storage%20potential%20of%20sugi%20forest&f=false
					co2vals.append('2.5')
				elif(p == 3):
					# Mixed Deciduous type: https://books.google.co.in/books?id=3EuFYjDdZhgC&pg=PA254&dq=carbon+storage+potential+of+deciduous+forest&hl=en&sa=X&ved=0ahUKEwjBnpTQ1cPMAhUDcI4KHRlhC1UQ6AEIKjAB#v=onepage&q=carbon%20storage%20potential%20of%20deciduous%20forest&f=false
					co2vals.append('0.14')
				else:
					# Other non-forest land
					co2vals.append('0')
			return co2vals


if __name__ == "__main__":
	f2co2 = Forest2CO2('training.csv')
	f2co2.train_data()
	with open('output.txt', 'w') as f:
		f.write('\n'.join(f2co2.test_data('testing.csv')))