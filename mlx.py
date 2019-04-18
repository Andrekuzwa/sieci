from mlxtend.data import iris_data
X, y = iris_data()

# standardize training data
X_std = (X - X.mean(axis=0)) / X.std(axis=0)

from mlxtend.classifier import MultiLayerPerceptron as MLP

nn1 = MLP(hidden_layers=[10],
          l2=0.00,
          l1=0.0,
          epochs=10000,
          eta=0.001,
          momentum=0.1,
          decrease_const=0.0,
          minibatches=1,
          random_seed=1,
          print_progress=3)

nn1 = nn1.fit(X_std, y)
print('\nAccuracy: %.2f%%' % (100 * nn1.score(X_std, y)))