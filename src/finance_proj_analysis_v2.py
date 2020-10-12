import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import shap
import seaborn as sns
import matplotlib.pyplot as plt

#Read Data
df = pd.read_csv('clean_no_drops.csv')

#Dropping rows without a date (means parsing didn't work and no data available)
df = df.dropna(subset = ['date_'])

#Dropping rows with 5+ NA's
df = df.dropna(thresh = 35)

#Define y variable
y_var = 'market_value_'

#Dropping rows with bad y
df = df.dropna(subset = [y_var])
df = df[df[y_var] != '0']
df = df[df[y_var] != '#DIV/0!']

#Drop columns unnecessary for analysis
df.drop(['Unnamed: 0',
         'company_name',
         'date_',
         'delta_dividends_', #too many NaN's
         'delta_market_value_forward_'], 
         axis = 1, inplace = True)

#Replace failed delta calculations with 0
cols = ['delta_comprehensive_income_','delta_revenue_','delta_stockholders_equity_','delta_market_value_']
df[cols] = df[cols].replace(to_replace = '#DIV/0!', value = 0)
for col in cols:
    df[col] = pd.to_numeric(df[col])

#X/Y Split
X = df.drop(y_var, axis = 1)
X_quant = X.loc[:,'total_comp_income_':'delta_market_value_']
X_nlp = X.loc[:,'doc2vec_dim_1':'doc2vec_dim_30']
Y = df[y_var]

#Train/Test Split (Whole Model)
X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size = 0.2,
                                                    random_state = 0)

#Train/Test Split (Quant Model)
X_q_train, X_q_test, Y_train, Y_test = train_test_split(X_quant,
                                                    Y,
                                                    test_size = 0.2,
                                                    random_state = 0)

#Train/Test Split (NLP Model)
X_n_train, X_n_test, Y_train, Y_test = train_test_split(X_nlp,
                                                    Y,
                                                    test_size = 0.2,
                                                    random_state = 0)

#Normalizing data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_q_train = scaler.fit_transform(X_q_train)
X_q_test = scaler.transform(X_q_test)

X_n_train = scaler.fit_transform(X_n_train)
X_n_test = scaler.transform(X_n_test)

X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
X_q_train = np.nan_to_num(X_q_train)
X_q_test = np.nan_to_num(X_q_test)

#Initialize model
model = RandomForestRegressor(n_estimators = 1000,
                                  max_depth = 3,
                                  random_state = 0)

#Entire model
model.fit(X_train, Y_train)
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)
print('Combined model training R2: {}'.format(train_r2.round(2)))
print('Combined model test R2: {}'.format(test_r2.round(2)))

#Just Quant
model.fit(X_q_train, Y_train)
Y_train_pred = model.predict(X_q_train)
Y_test_pred = model.predict(X_q_test)
train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)
print('Quant. model training R2: {}'.format(train_r2.round(2)))
print('Quant. model test R2: {}'.format(test_r2.round(2)))

#Just NLP
model.fit(X_n_train, Y_train)
Y_train_pred = model.predict(X_n_train)
Y_test_pred = model.predict(X_n_test)
train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)
print('NLP model training R2: {}'.format(train_r2.round(2)))
print('NLP model test R2: {}'.format(test_r2.round(2)))

# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_n_test)

# p = shap.summary_plot(shap_values, X_n_test, show=False, matplotlib = True) 
# display(p)