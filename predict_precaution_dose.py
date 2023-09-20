import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('covid19_data.csv')

train_df, test_df = train_test_split(df, test_size=0.2)

X_train = train_df[['dose1', 'dose2', 'dose3', 'population']]
y_train = train_df['precaution_dose']
X_test = test_df[['dose1', 'dose2', 'dose3', 'population']]
y_test = test_df['precaution_dose']

model = LinearRegression()
model.fit(X_train, y_train)

r_sq = model.score(X_test, y_test)
print(f"\nCoefficient of Determination(R^2): {r_sq}\n")

y_pred = model.predict(X_test)
result_df = pd.DataFrame(
    {'state': test_df['state'], 'precaution_dose': y_test, 'predicted_precaution_dose': y_pred.astype(int)})
print(result_df)

plt.scatter(result_df['state'], y_test, color='red')
plt.plot(result_df['state'], y_pred.astype(int), color='blue')
plt.title('Multi-variate Linear Regression on Covid19 State Data')
plt.xlabel('State')
plt.ylabel('Precaution Doses')
plt.show()
