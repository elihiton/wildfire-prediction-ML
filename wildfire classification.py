import requests
import numpy as np
import io
import pandas as pd

url = 'https://github.com/ouladsayadyounes/WildFires/raw/master/WildFires_DataSet.csv'
download = requests.get(url).content
wildfire_df = pd.read_csv(io.StringIO(download.decode('utf-8')))

#split dataframe into features and classes
wildfire_df_X = wildfire_df.iloc[:,:3]
wildfire_df_y = wildfire_df.iloc[:,3:]


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score as precision, recall_score as recall, accuracy_score as acc


folds = 6
kf = KFold(n_splits = folds, shuffle = True, random_state = 0)
neighbors = 18 #18 neighbors chosen through hyperparameter testing
knn = KNeighborsClassifier(n_neighbors = neighbors, weights = 'distance') #distance chosen through hyperparameter testing
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
    sum_f += f_beta
    sum_acc += acc(working_test_y, pred)
acc_avg = sum_acc/folds
f_beta_avg = sum_f/folds
print(f"Average weighted f_2 for 'fire' = {f_beta_avg}")
#prints Average weighted f_2 for 'fire' = 0.4038476588861058
print(f"Average accuracy = {acc_avg}")
#prints Average accuracy = 0.8190426532531796


#Create and train a random forest (seems appropriate given what we're investigating)
from sklearn.ensemble import RandomForestClassifier
trees = 330 #chosen through hyperparameter testing
sum_f = 0
sum_acc = 0
rf = RandomForestClassifier(n_estimators=trees, max_features=2, random_state=0)
for i, (train_index, test_index) in enumerate(kf.split(wildfire_df_X)):
    working_train_X = wildfire_df_X.iloc[train_index]
    working_train_y = wildfire_df_y.iloc[train_index]
    working_test_X = wildfire_df_X.iloc[test_index]
    working_test_y = wildfire_df_y.iloc[test_index]
    rf.fit(working_train_X, working_train_y.values.ravel())
    pred_rf=rf.predict(working_test_X)
    ps = precision(working_test_y, pred_rf, pos_label = 'fire')
    rs = recall(working_test_y, pred_rf, pos_label = 'fire')
    f_beta = (1+beta**2)*ps*rs/(beta**2*ps+rs)
    sum_f += f_beta
    sum_acc += acc(working_test_y, pred_rf)
acc_avg = sum_acc/folds
f_beta_avg = sum_f/folds
print(f"Average weighted f_2 for 'fire' = {f_beta_avg}")
#prints Average weighted f_2 for 'fire' = 0.5198478937874957
print(f"Average accuracy = {acc_avg}")
#prints Average accuracy = 0.8371242792295425


#re-running knn on a more balanced dataset
from sklearn.utils import resample
kf = KFold(n_splits = folds, shuffle = True, random_state = 0)
knn = KNeighborsClassifier(n_neighbors = neighbors, weights = 'distance') #re-initialize classifier
downsample_rate = .7
sum_f = 0
sum_acc = 0 #re-initialize metrics
for i, (train_index, test_index) in enumerate(kf.split(wildfire_df_X)):
    working_train = wildfire_df.iloc[train_index]
    working_test = wildfire_df.iloc[test_index]
    train_group = working_train.groupby(wildfire_df.CLASS)
    train_fire = train_group.get_group("fire")
    train_nf = train_group.get_group("no_fire")
    train_nf_downsampled = resample(train_nf, replace=False, n_samples=round(len(train_nf)*downsample_rate), random_state = 0) #shrink majority class
    working_train = pd.concat([train_fire, train_nf_downsampled]) #re-assemble training data
    working_train_X = working_train.iloc[:,:3]
    working_train_y = working_train.iloc[:,3:]
    working_test_X = working_test.iloc[:,:3]
    working_test_y = working_test.iloc[:,3:]
    knn.fit(working_train_X, working_train_y.values.ravel())
    pred=knn.predict(working_test_X)
    ps = precision(working_test_y, pred, pos_label = 'fire')
    rs = recall(working_test_y, pred, pos_label = 'fire')
    f_beta = (1+beta**2)*ps*rs/(beta**2*ps+rs)
    sum_f += f_beta
    sum_acc += acc(working_test_y, pred)
acc_avg = sum_acc/folds
f_beta_avg = sum_f/folds
print(f"Average weighted f_2 score for 'fire' over knn = {f_beta_avg}")
#prints Average weighted f1 for 'fire' = 0.45799357647183725
print(f"Average accuracy over knn = {acc_avg}")
#prints Average accuracy = 0.7804850120639596


#re-running random forest with more balanced classes
sum_f = 0
sum_acc = 0 #reinitialize metrics
rf = RandomForestClassifier(n_estimators=trees, max_features=2, random_state=0) #reinitialize classifier
for i, (train_index, test_index) in enumerate(kf.split(wildfire_df_X)):
    working_train = wildfire_df.iloc[train_index]
    working_test = wildfire_df.iloc[test_index]
    train_group = working_train.groupby(wildfire_df.CLASS)
    train_fire = train_group.get_group("fire")
    train_nf = train_group.get_group("no_fire")
    train_nf_downsampled = resample(train_nf, replace=False, n_samples=round(len(train_nf)*downsample_rate), random_state = 0)
    working_train = pd.concat([train_fire, train_nf_downsampled])
    working_train_X = working_train.iloc[:,:3]
    working_train_y = working_train.iloc[:,3:]
    working_test_X = working_test.iloc[:,:3]
    working_test_y = working_test.iloc[:,3:]
    rf.fit(working_train_X, working_train_y.values.ravel())
    pred_rf=rf.predict(working_test_X)
    ps = precision(working_test_y, pred_rf, pos_label = 'fire')
    rs = recall(working_test_y, pred_rf, pos_label = 'fire')
    f_beta = (1+beta**2)*ps*rs/(beta**2*ps+rs)
    sum_f += f_beta
    sum_acc += acc(working_test_y, pred_rf)
acc_avg = sum_acc/folds
f_beta_avg = sum_f/folds
print(f"Average weighted f_2 for 'fire' over rf = {f_beta_avg}")
#prints Average weighted f_2 for 'fire' = 0.5834531412212446
print(f"Average accuracy over rf= {acc_avg}")
#prints Average accuracy = 0.8272113851061219
