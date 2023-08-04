import requests
import numpy as np
import io
import pandas as pd
import sklearn

url = 'https://github.com/ouladsayadyounes/WildFires/raw/master/WildFires_DataSet.csv'
download = requests.get(url).content
wildfire_df = pd.read_csv(io.StringIO(download.decode('utf-8')))

print (wildfire_df.head())

#split dataframe into parameters and classes
wildfire_df_X = wildfire_df.iloc[:,:3]
wildfire_df_y = wildfire_df.iloc[:,3:]
