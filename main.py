import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# reading csv files
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

plt.hist(train_data['SalePrice'])
plt.xlabel("$")
plt.ylabel("Count")
plt.title("SalePrice")
# plt.show()

train_data['SalePrice'] = np.log1p(train_data['SalePrice'])

plt.hist(train_data['SalePrice'])
plt.xlabel("log1($)")
plt.ylabel("Count")
plt.title("Sale Price Post-Transform")
# plt.show()

list(train_data.columns)

# train_data.describe()

# getting value of colls
# for column in train_data.columns:
    # print("\n---- %s ---" % column)
    # print(train_data[column].value_counts())


# object cols
cat_feat = train_data.select_dtypes(include=[np.object])
# cat_feat.info()
cat_feat.nunique()

# numerical cols
num_category_base = train_data.select_dtypes(include=[np.number])
# num_category_base.info()

# we are getting onfo about columns
for col in num_category_base.columns:
    plt.hist(num_category_base[col])
    # plt.show()

# first view on corr map
corr_matrix = num_category_base.corr()
plt.figure(figsize=(40, 40))
data_map = sns.heatmap(corr_matrix)
plt.title('Correlation Matrix of features', fontsize=20)
# plt.show()

# look for corrs
# print(corr_matrix[['SalePrice']].sort_values(['SalePrice'], ascending = False))

colls_miss = [col for col in train_data.columns if train_data[col].isnull().any()]

# look for missing colls
# msno.matrix(train_data[colls_miss])

for col_name in colls_miss:
    # print(f'{col_name}:\n{train_data[col_name].unique()}\n')
    isna_val = train_data[col_name].isna().sum()
    count_of_col = train_data[col_name].count()
    # print(f'isna: {isna_val}\n')
    # print(f'count: {count_of_col}\n')
    # print(f'%: {isna_val / (isna_val + count_of_col) * 100}\n\n----------------------------------------\n')

# save testId
test_Id = test_data['Id']

redun = ['GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars']
useless = ['YrSold','MoSold', 'Id']
sparse = ['PoolQC', 'MiscFeature', 'Alley']

train_data.drop(redun, axis=1, inplace=True)
test_data.drop(redun, axis=1, inplace=True)

train_data.drop(useless, axis = 1, inplace = True)
test_data.drop(useless, axis = 1, inplace = True)

train_data.drop(sparse, axis = 1, inplace = True)
test_data.drop(sparse, axis = 1, inplace = True)

num_category_base = list(set(num_category_base)-set(redun)-set(useless)-set(sparse))
cat_feat = list(set(cat_feat)-set(redun)-set(useless)-set(sparse))

train_data = train_data[train_data['GrLivArea'] < 4500]

num_category_base.remove('SalePrice')

X_cols = num_category_base + cat_feat
numerical_transformer = SimpleImputer(strategy='constant', fill_value = 0)

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_category_base),
        ('cat', categorical_transformer, cat_feat)
    ])

Y_train = train_data['SalePrice']
X_train = train_data[X_cols]
X_test = test_data[X_cols]

model = XGBRegressor(n_estimators = 3460,
                     max_depth = 3,
                     learning_rate = 0.01,
                     subsample = 0.7,
                     seed=1,
                     early_stopping_rounds=5,
                     eval_set=[(X_train, Y_train)],
                     verbose=False)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

my_pipeline.fit(X_train, Y_train)

preds = my_pipeline.predict(X_test)

output = pd.DataFrame({'Id': test_Id,
                       'SalePrice': preds})
output["SalePrice"] = np.expm1(output["SalePrice"])
output.to_csv('submission.csv', index=False)
