import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_dataset(filename):
	features = []
	target = []
	target_names = set()

	with open(filename) as f:
		for line in f:
			tokens = line.strip().split('\t')
			features.append([float(tk) for tk in tokens[:-1]])
			target.append(tokens[-1])
			target_names.add(tokens[-1])

	features = np.array(features)
	target_names = list(target_names)
	target_names.sort()
	target = np.array([target_names.index(t) for t in target])

	return {
		'features': features,
		'target_names' : target_names,
		'target': target,
	}

data = load_dataset('seeds.tsv')
features = data['features']
target = data['target']

knn = KNeighborsClassifier(n_neighbors=5)
classifier = Pipeline([('norm', StandardScaler()), ('knn', knn)])

kf = model_selection.KFold(n_splits=5, shuffle=False)

means = []
for training, testing in kf.split(features):
	classifier.fit(features[training], target[training])
	prediction = classifier.predict(features[testing])
	
	curmean = np.mean(prediction == target[testing])
	means.append(curmean)

print('Mean of accuracy: {:.1%}'.format(np.mean(means)))

