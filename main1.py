import warnings
from imblearn.over_sampling import SMOTE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from scipy import ndimage
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('exoTrain.csv')
datatest = pd.read_csv('exoTest.csv')
print(dataset.shape)
print(dataset.head())
print(dataset.info())
print(dataset.describe())

print(dataset.isnull().sum())

dset_null = round(100 * (dataset.isnull().sum()) / len(dataset), 2)
print(dset_null)

print(dataset.shape)
dataset = dataset.dropna()
print(dataset.shape)

n = dataset['LABEL'].value_counts()
print(n)

n.plot(kind='bar')
plt.title('Count of each Planet Type Present in Dataset \n 1=not Exoplanet|| 2=Exoplanet')
plt.xlabel('Planet Type')
plt.ylabel('Number of each planet')
plt.legend()
plt.show()
"""
plt.figure(figsize=(10,8))
plt.title('Distribution of flux values', fontsize=15)
plt.xlabel('Flux values')
plt.ylabel('Flux intensity')
plt.plot(dataset.iloc[0,])
plt.plot(dataset.iloc[1,])
plt.plot(dataset.iloc[2,])
plt.plot(dataset.iloc[3,])
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

labels_1=[100,200,300]
for i in labels_1:
    plt.figure(figsize=(3,3))
    plt.hist(dataset.iloc[i,:], bins=200)
    plt.title("Gaussian Histogram")
    plt.xlabel("Flux values")
    plt.show()

labels_1=[16,21,25]
for i in labels_1:
    plt.figure(figsize=(3,3))
    plt.hist(dataset.iloc[i,:], bins=200)
    plt.title("Gaussian Histogram")
    plt.xlabel("Flux values")
    plt.show()

l = ['1', '2', '3', '4', '5']
for i in range(len(l)):
    sns.boxplot(dataset['FLUX.' + l[i]], whis=2)
    plt.title('Box Plot')
    plt.xlabel('Values')
    plt.ylabel('Boxplot')
    plt.show()
"""
def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range

columns = ['FLUX.1', 'FLUX.2', 'FLUX.3', 'FLUX.4', 'FLUX.5']
for column in dataset[columns].columns:
    lr,ur=remove_outlier(dataset[column])
    dataset[column]=np.where(dataset[column]>ur,ur,dataset[column])
    dataset[column]=np.where(dataset[column]<lr,lr,dataset[column])

for j in columns:
    plt.figure(figsize=(7.5, 4.5))
    print(j)
    sns.boxplot(dataset[j],whis=1.5)
    plt.show()

print(dataset.shape)

X_train = dataset.drop(['LABEL'], axis=1)
Y_train = dataset['LABEL']
X_test = datatest.drop(['LABEL'], axis=1)
Y_test = datatest['LABEL']

X_train = normalize(X_train)
X_test = normalize(X_test)

X_train = ndimage.gaussian_filter(X_train, sigma=10)
X_test = ndimage.gaussian_filter(X_test, sigma=10)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform((X_test))

print(dataset)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

k_cl = KNeighborsClassifier().fit(X_train, Y_train)
predicted_data = k_cl.predict(X_test)
print(predicted_data)
print(classification_report(Y_test, predicted_data))
cm = confusion_matrix(Y_test, predicted_data)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(predicted_data, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

prob_pos = k_cl.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, pos_label=2, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc='best')
plt.show()

l = LogisticRegression().fit(X_train, Y_train)
predicted_data = l.predict(X_test)
print(predicted_data)
print(classification_report(Y_test, predicted_data))
cm = confusion_matrix(Y_test, predicted_data)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(predicted_data, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

prob_pos = l.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, pos_label=2, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc='best')
plt.show()

d_tree = DecisionTreeClassifier().fit(X_train, Y_train)
predicted_data = d_tree.predict(X_test)
print(predicted_data)
print(classification_report(Y_test, predicted_data))
cm = confusion_matrix(Y_test, predicted_data)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(predicted_data, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

prob_pos = d_tree.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, pos_label=2, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc='best')
plt.show()

gbc = GradientBoostingClassifier(learning_rate=0.05, random_state=100, max_features=5).fit(X_train, Y_train)
predicted_data = gbc.predict(X_test)
print(predicted_data)
print(classification_report(Y_test, predicted_data))
cm = confusion_matrix(Y_test, predicted_data)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(predicted_data, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

prob_pos = GradientBoostingClassifier.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, pos_label=2, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc='best')
plt.show()

model = SMOTE()
print(dataset['LABEL'].value_counts())
o_train_x, o_train_y = model.fit_resample(dataset.drop(['LABEL'], axis=1), dataset['LABEL'])
o_train_y = o_train_y.astype('int')

o_train_y.value_counts().plot(kind='bar', x='Index', y='Label')
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(o_train_x, o_train_y, test_size=0.25, random_state=0)

k_cl.fit(X_train, Y_train)
predicted_data = k_cl.predict(X_test)
print(predicted_data)
print(classification_report(Y_test, predicted_data))
cm = confusion_matrix(Y_test, predicted_data)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(predicted_data, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

prob_pos = k_cl.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, pos_label=2, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc='best')
plt.show()

l.fit(X_train, Y_train)
predicted_data = l.predict(X_test)
print(predicted_data)
print(classification_report(Y_test, predicted_data))
cm = confusion_matrix(Y_test, predicted_data)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(predicted_data, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

prob_pos = l.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, pos_label=2, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc='best')
plt.show()

d_tree.fit(X_train, Y_train)
predicted_data = d_tree.predict(X_test)
print(predicted_data)
print(classification_report(Y_test, predicted_data))
cm = confusion_matrix(Y_test, predicted_data)
print(cm)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(predicted_data, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

prob_pos = d_tree.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, pos_label=2, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc='best')
plt.show()

gbc.fit(X_train, Y_train)
predicted_data = gbc.predict(X_test)
print(predicted_data)
print(classification_report(Y_test, predicted_data))
cm = confusion_matrix(Y_test, predicted_data)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(10, 6))
sns.kdeplot(Y_test, label='True')
sns.kdeplot(predicted_data, label='Predicted')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.title('Probability Distribution of Original vs Predicted Classifications')
plt.legend()
plt.show()

prob_pos = GradientBoostingClassifier.predict_proba(X_test)[:, 1]

# Plot calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, prob_pos, pos_label=2, n_bins=10)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, 's-', label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve')
plt.legend(loc='best')
plt.show()
