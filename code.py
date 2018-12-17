import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from scipy import ndimage

from methods import *

class_size = 8

def get_features(image, mode='train'):
	# hisogram equalization
	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
	image[:, :, 0] = clahe.apply(image[:, :, 0])
	image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
	
	
	# if mode == 'test':
	# 	plt.imshow(image[:,:,::-1])
	# 	plt.show()
	means, stds = cv2.meanStdDev(image)
	stats = np.concatenate([means, stds]).flatten()
	h, w, _ = image.shape
	if h > w:
		image = ndimage.rotate(image, 90)
	image = cv2.resize(image, (500, 400), interpolation=cv2.INTER_CUBIC)
	image = image[100:300, 100:400]
	for i in range(0, 200, 50):
		for j in range(0, 300, 50):
			hist = cv2.calcHist([image[i:i+50, j:j+50]], [0, 1, 2], None, [5, 5, 5], [0, 256, 0, 256, 0, 256]).flatten()
			stats = np.concatenate([stats, hist]).flatten()
	return stats


def plot(X, Y, labels=range(class_size)):
	colors = plt.cm.rainbow(np.linspace(0, 1, class_size))
	for label, color in zip(labels, colors):
		plt.scatter(X[:, 0].real[Y == label], X[:, 1].real[Y == label], color=color, label=label)
	plt.legend()


def load_data():
	print('Loading images...')
	x_train = []
	y_train = []
	class_names = {}
	for label, folder in enumerate(os.listdir('images')):
		print(folder, 'images')
		color_path = os.path.join('images', folder)
		class_names[label] = folder
		for file in os.listdir(color_path):
			image_path = os.path.join(color_path, file)
			image = cv2.imread(image_path)
			x_train.append(image)
			y_train.append(label)
	print('Done!')
	return np.array(x_train), np.array(y_train), class_names

x_train, y_train, class_names = load_data()
features = np.array([get_features(img) for img in x_train])

df = pd.DataFrame(features, columns=['mean_' + i for i in ['blue', 'green', 'red']] + ['std_' + i for i in ['blue', 'green', 'red']] + ['hist_' + str(i) for i in range(3000)])
# df['label'] = y_train

# Box plot
# df.loc[:, 'hist_0':'hist_5'].plot(kind='box', subplots=True, layout=(2,3), sharex=False, sharey=False, figsize=(9,9), title='Box Plot for each input variable')
# plt.plot()
# plt.show()

# histogram
# import pylab as pl
# df.loc[:, 'mean_blue':'std_red'].hist(bins=30, figsize=(9,9))
# pl.suptitle("Histogram for each mean and variance")
# plt.show()

# Printing scatter matrix
from matplotlib import cm
X = df.loc[:, 'mean_blue':'std_red']
y = y_train
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X, c = y, marker = 'o', s=20, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
plt.suptitle('Scatter-matrix for each input variable')
plt.savefig('scatter_matrix.png')
plt.show()

# Scaling the training set
print('Preprocessing')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

print('Training')
mode, model = knn(features, y_train)

test_features = []
test_names = []
for folder in os.listdir('test'):
	print(f'Testing on {folder} folder')
	color_path = os.path.join('test', folder)
	for file in os.listdir(color_path):
		image_path = os.path.join(color_path, file)
		test_names.append(image_path)
		test_features.append(get_features(cv2.imread(image_path), 'test'))

test_features = scaler.transform(test_features)
response = model.predict(test_features)
neighbors = model.kneighbors_graph(test_features).indices.reshape((-1, 5))
del test_features

# Calculating the accuracy
acc = 0
for name, cls in zip(test_names, list(response.flatten())):
	[folder, n] = name.split('\\')[-2:]
	print('{:25s}: {}'.format(n, class_names[cls]))
	if class_names[cls] == folder:
		acc += 1
acc /= len(test_names)
print(f'Accuracy on testing set: {acc}')

# Plotting the results and neighbors
for name, cls, ns in zip(test_names, list(response.flatten()), neighbors):
	plt.suptitle(f'Mode: {mode}')
	gs = GridSpec(3, 4)
	axs1 = plt.subplot(gs[:, :2])
	axs1.set_title(f'Predict: {class_names[cls]}')
	axs1.imshow(cv2.imread(name)[:, :, ::-1])
	
	axs = [plt.subplot(gs[0, 2]), plt.subplot(gs[0, 3]), plt.subplot(gs[1, 2]),
	       plt.subplot(gs[1, 3]), plt.subplot(gs[2, 2:4])]
	
	for n, ax in zip(ns, axs):
		ax.set_title(class_names[y_train[n]])
		ax.imshow(x_train[n][:,:,::-1])
		ax.set_xticklabels([])
		ax.set_yticklabels([])
	mng = plt.get_current_fig_manager()
	mng.full_screen_toggle()
	plt.show()
