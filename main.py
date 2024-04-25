import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

dataset = pd.read_csv('exoplanet.csv')

print(dataset.head())
print(dataset.info())
print(dataset.describe())

cols = ['distance', 'stellar_magnitude', 'radius_multiplier', 'orbital_radius', 'orbital_period', 'eccentricity']
print(dataset[cols].isnull().sum())

dset_null = round(100 * (dataset.isnull().sum()) / len(dataset), 2)
print(dset_null)

dataset = dataset.dropna()
print(dataset.shape)

col = ['planet_type', 'mass_wrt', 'radius_wrt']
for i in col:
    n = dataset[i].value_counts()
    print(n)
    n.plot(kind='bar')
    plt.title('Bar Graph Analysis: - ' + i)
    plt.xlabel(i)
    plt.ylabel('Count')
    plt.legend()
    plt.show()

n = dataset['discovery_year'].value_counts()
n = n.sort_index()
print(n)
n.plot(kind='line')
plt.title('Trend Analysis - Discovery Year')
plt.xlabel('Discovery Year')
plt.ylabel('Count')
plt.legend()
plt.show()

n = dataset['eccentricity'].value_counts()
n = n.sort_index()
print(n)
n.plot(kind='line')
plt.title('Trend Analysis - Eccentricity')
plt.xlabel('Eccentricity')
plt.ylabel('Count')
plt.legend()
plt.show()

selected_columns = dataset.columns[1:-1]
selected_df = dataset[selected_columns]

numeric_columns = selected_df.select_dtypes(include=['number']).columns
numeric_df = selected_df[numeric_columns]

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(selected_df['stellar_magnitude'], kde=True, bins=20, color='skyblue')
plt.title('Stellar Magnitude Histogram')
plt.xlabel('Stellar Magnitude')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(selected_df['distance'], kde=True, bins=20, color='skyblue')
plt.title('Distance Analysis Histogram')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()

columns = ['distance', 'stellar_magnitude']

for i in columns:
    sns.boxplot(dataset[i], whis=2)
    plt.title(i + ' Box Plot')
    plt.xlabel('Values')
    plt.ylabel('Boxplot')
    plt.show()

    sns.violinplot(dataset[i], color='skyblue')
    plt.title(i + ' Violin Plot')
    plt.xlabel('Values')
    plt.ylabel('Violin plot')
    plt.show()


X = dataset[
    ['distance', 'stellar_magnitude', 'mass_multiplier', 'radius_multiplier', 'orbital_radius',
     'orbital_period', 'eccentricity']]
print('Independent Features')
print(X)
Y = dataset[['planet_type']]
Y = np.array(Y)
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)
print('Dependent Features')
print(Y.ravel())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
print('Training Independent Features')
print(X_train)
print('Testing Independent Features')
print(X_test)
print('Training Dependent Features')
print(Y_train)
print('Testing Dependent Features')
print(Y_test)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier().fit(X_train, Y_train)
y_pred1 = classifier.predict(X_test)
print(np.array(y_pred1))
print(classification_report(Y_test, y_pred1))
cm = confusion_matrix(Y_test, y_pred1)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(y_pred1, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

l = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, Y_train)
y_pred2 = l.predict(X_test)
print(y_pred2)
print(classification_report(Y_test, y_pred2))
cm = confusion_matrix(Y_test, y_pred2)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(y_pred2, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

t = DecisionTreeClassifier().fit(X_train, Y_train)
y_pred3 = t.predict(X_test)
print(y_pred3)
print(classification_report(Y_test, y_pred3))
cm = confusion_matrix(Y_test, y_pred3)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(y_pred3, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

gbc = GradientBoostingClassifier(learning_rate=0.05, random_state=100, max_features=5).fit(X_train, Y_train)
y_pred4 = gbc.predict(X_test)
print(y_pred4)
print(classification_report(Y_test, y_pred4))
cm = confusion_matrix(Y_test, y_pred4)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(y_pred4, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

name = input('Enter the name of the exoplanet: - ')
dist = float(input('Enter the distance: - '))
stmg = float(input('Enter the Stellar Magnitude: - '))
di_year = int(input('Enter the discovery year: - '))
m_multi = float(input('Enter the mass multiplier: - '))
m_wrt = input('Enter the mass wrt: - ')
r_multi = float(input('Enter the radius multiplier: - '))
r_wrt = input('Enter the radius wrt: - ')
o_radius = float(input('Enter the orbital radius: - '))
o_period = float(input('Enter the orbital period: - '))
ec = float(input('Enter the eccentricity: - '))
method = input('Enter the method: - ')

new_data = np.array([dist, stmg, m_multi, r_multi, o_radius, o_period, ec])
new_data = new_data.reshape(1, -1)
predicted = t.predict(new_data)
decoded_predictions = encoder.inverse_transform(predicted)
print(decoded_predictions)
