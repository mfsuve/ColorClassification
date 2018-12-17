def logistic_regression(X_train, y_train):
	from sklearn.linear_model import LogisticRegression
	logreg = LogisticRegression()
	logreg.fit(X_train, y_train)
	print('Accuracy of Logistic regression classifier on training set: {:.2f}'
	      .format(logreg.score(X_train, y_train)))
	# print('Accuracy of Logistic regression classifier on test set: {:.2f}'
# 	#       .format(logreg.score(X_test, y_test)))
	return 'logreg', logreg

def decision_tree(X_train, y_train):
	from sklearn.tree import DecisionTreeClassifier
	clf = DecisionTreeClassifier().fit(X_train, y_train)
	print('Accuracy of Decision Tree classifier on training set: {:.2f}'
	      .format(clf.score(X_train, y_train)))
	# print('Accuracy of Decision Tree classifier on test set: {:.2f}'
	#       .format(clf.score(X_test, y_test)))
	return 'tree', clf

def knn(X_train, y_train):
	from sklearn.neighbors import KNeighborsClassifier
	clf = KNeighborsClassifier()
	clf.fit(X_train, y_train)
	print('Accuracy of K-NN classifier on training set: {:.2f}'
	      .format(clf.score(X_train, y_train)))
	# print('Accuracy of K-NN classifier on test set: {:.2f}'
	#       .format(knn.score(X_test, y_test)))
	return 'knn', clf

def lda(X_train, y_train):
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
	clf = LinearDiscriminantAnalysis()
	clf.fit(X_train, y_train)
	print('Accuracy of LDA classifier on training set: {:.2f}'
	      .format(clf.score(X_train, y_train)))
	# print('Accuracy of LDA classifier on test set: {:.2f}'
	#       .format(lda.score(X_test, y_test)))
	return 'lda', clf

def gaussian_naive(X_train, y_train):
	from sklearn.naive_bayes import GaussianNB
	gnb = GaussianNB()
	gnb.fit(X_train, y_train)
	print('Accuracy of GNB classifier on training set: {:.2f}'
	      .format(gnb.score(X_train, y_train)))
	# print('Accuracy of GNB classifier on test set: {:.2f}'
	#       .format(gnb.score(X_test, y_test)))
	return 'gaussian', gnb

def svc(X_train, y_train):
	from sklearn.svm import SVC
	svm = SVC()
	svm.fit(X_train, y_train)
	print('Accuracy of SVM classifier on training set: {:.2f}'
	      .format(svm.score(X_train, y_train)))
	# print('Accuracy of SVM classifier on test set: {:.2f}'
	#       .format(svm.score(X_test, y_test)))
	return 'SVM', svm