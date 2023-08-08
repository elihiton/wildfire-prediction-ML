import requests
import numpy as np
import io
import pandas as pd

url = 'https://github.com/ouladsayadyounes/WildFires/raw/master/WildFires_DataSet.csv'
download = requests.get(url).content
wildfire_df = pd.read_csv(io.StringIO(download.decode('utf-8')))

#split dataframe into features  and classes
wildfire_df_X = wildfire_df.iloc[:,:3]
wildfire_df_y = wildfire_df.iloc[:,3:]


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score as precision, recall_score as recall, accuracy_score as acc


folds = 6
kf = KFold(n_splits = folds, shuffle = True, random_state = 0)

knn = KNeighborsClassifier(n_neighbors = neighbors, weights = 'distance') #distance chosen through hyperparameter testing
neighbors = 18 #18 neighbors chosen through hyperparameter testing


sum_f = 0
sum_acc = 0
beta = 2 #beta > 1 weights recall more heavily, beta < 1 weights precision more heavily


for i, (train_index, test_index) in enumerate(kf.split(wildfire_df_X)):

    working_train_X = wildfire_df_X.iloc[train_index]
    working_train_y = wildfire_df_y.iloc[train_index]

    working_test_X = wildfire_df_X.iloc[test_index]
    working_test_y = wildfire_df_y.iloc[test_index]

    knn.fit(working_train_X, working_train_y.values.ravel())
    pred=knn.predict(working_test_X)
    ps = precision(working_test_y, pred, pos_label = 'fire')
    rs = recall(working_test_y, pred, pos_label = 'fire')
    f_beta = (1+beta**2)*ps*rs/(beta**2*ps+rs)
    #print(f"Fold {i}: {f_beta}")
    sum_f += f_beta
    sum_acc += acc(working_test_y, pred)

acc_avg = sum_acc/folds
f_beta_avg = sum_f/folds
print(f"Average weighted f1 for 'fire' = {f_beta_avg}")
#prints Average weighted f1 for 'fire' = 0.4038476588861058
print(f"Average accuracy = {acc_avg}")
#prints Average accuracy = 0.8190426532531796
