from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
# import seabornas sns
import sklearn

# Lily
# data = pd.read_csv("/Users/lilyirvin/Downloads/DFL/primary_walk_vs_phones.csv")

# Collin
data = pd.read_csv("../resources/primary_walk_vs_phones.csv")

# print(data.head(10))
# print("\n\n")
# print(data.tail(10))
# print(data.shape)

# sns.pairplot(data, x_vars=['walkcontacts', 'phonecontacts'], y_vars=['voted'])

X = data[['walkcontacts', 'phonecontacts']]
Y = data[['voted']]

#comment to test

# print(X.head(5))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

linreg = LinearRegression()
linreg.fit(X_train, Y_train)

print(linreg.intercept_)
print(linreg.coef_)
