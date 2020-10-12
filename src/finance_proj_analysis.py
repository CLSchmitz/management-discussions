import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#Read Data
df = pd.read_csv('clean_data_with_nlp.csv')

#Drop columns unnecessary for analysis
df.drop(['Unnamed: 0',
         'company_name',
         'date_',
         'delta_dividends_'], #too many NaN's
         axis = 1, inplace = True)

#Replace failed delta calculations and very large values with NA's
cols = ['delta_comprehensive_income_','delta_revenue_','delta_stockholders_equity_','delta_market_value_']
df[cols] = df[cols].replace(to_replace = ['0','#DIV/0!', 0], value = np.nan)
for col in cols:
    df[col] = pd.to_numeric(df[col])
df[cols] = df[cols].apply(lambda x: [y if abs(y) <= 30 else np.nan for y in x])

#X/Y Split
X = df.drop('delta_market_value_forward_', axis = 1)
X_quant = X.loc[:,'total_comp_income_':'delta_market_value_']
X_nlp = X.loc[:,'doc2vec_dim_1':'doc2vec_dim_100']
Y = df['delta_market_value_forward_']

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

#Running Regression
model = LinearRegression()

#Entire model
model.fit(X_train, Y_train)
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)
train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)
print(train_r2, test_r2)

#Just Quant
model.fit(X_q_train, Y_train)
Y_train_pred = model.predict(X_q_train)
Y_test_pred = model.predict(X_q_test)
train_r2 = r2_score(Y_train, Y_train_pred)
test_r2 = r2_score(Y_test, Y_test_pred)
print(train_r2, test_r2)

#Just NLP