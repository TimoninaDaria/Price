from scipy import sparse
import numpy
import pandas
import sklearn.ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
train = pandas.read_csv('Train.csv')
test = pandas.read_csv('Test.csv')

train = train.replace(numpy.nan, -999)
test = test.replace(numpy.nan, -999)

COLUMNS = ['street_id', 'build_tech', 'floor', 'area', 'rooms', 'balcon', 'metro_dist', 'g_lift', 'n_photos', 'kw1', 'kw2', 'kw3', 'kw4', 'kw5', 'kw6', 'kw7', 'kw8', 'kw9', 'kw10', 'kw11', 'kw12', 'kw13']

Y = train['price'].values
X = train[COLUMNS].values
X_test = test[COLUMNS].values
X_train_date =  pandas.to_datetime(train['date']).dt.year.values
X_test_date =  pandas.to_datetime(test['date']).dt.year.values
X_train_date_new = []

for i in X_train_date:
    X_train_date_new.append([i])

X_test_date_new = []
for i in X_test_date:
    X_test_date_new.append([i])

X = numpy.append(X, numpy.asarray(X_train_date_new), axis = 1)
X_test =  numpy.append(X_test , numpy.asarray(X_test_date_new), axis = 1)

		
mdl = sklearn.ensemble.GradientBoostingRegressor(n_estimators = 1000, learning_rate= 0.085,  min_samples_leaf = 2, max_depth=6, min_samples_split=5)		

mdl.fit(X, Y)

Y_preds = mdl.predict(X_test)


test['price'] = Y_preds

test[['id', 'price']].to_csv('sub.csv', index=False)
