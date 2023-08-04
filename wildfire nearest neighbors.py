import requests
import numpy as np
import io
import pandas as pd
import sklearn

url = 'https://github.com/ouladsayadyounes/WildFires/raw/master/WildFires_DataSet.csv'
download = requests.get(url).content
wildfire_df = pd.read_csv(io.StringIO(download.decode('utf-8')))

#split dataframe into parameters and classes
wildfire_df_X = wildfire_df.iloc[:,:3]
wildfire_df_y = wildfire_df.iloc[:,3:]

# Split into train and test data
#can come back later and do k-folds later
np.random.seed(1)
indices = np.random.permutation(len(wildfire_df_X))
wildfire_df_X_train = wildfire_df_X.iloc[indices[:-300]]
wildfire_df_y_train = wildfire_df_y.iloc[indices[:-300]]
wildfire_df_X_test = wildfire_df_X.iloc[indices[-300:]]
wildfire_df_y_test = wildfire_df_y.iloc[indices[-300:]]

# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20, weights='distance')#maybe experiment with lower k
knn.fit(wildfire_df_X_train, wildfire_df_y_train.values.ravel())
pred=knn.predict(wildfire_df_X_test)

#accuracy and metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(classification_report(wildfire_df_y_test, pred))
print(confusion_matrix(wildfire_df_y_test, pred))
