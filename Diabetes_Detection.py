import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cProfile import label
from nltk.corpus.reader.ieer import titles

diabetes = pd.read_csv("C:/Users/ksaldanh/Documents/Diabetes.csv", encoding = 'unicode_escape')
print("Dimensions of the diabetes dataset is {}".format(diabetes.shape))

#In the dataset we are interested in the outcome variable, Here 0 means no diabetes and 1 means diabtes,
#so lets counts the number of 0s and 1s 
#print(diabetes.groupby("Outcome").size())

#EXPLORATORY DATA ANALYSIS(EDA)
#Lets visualize it because pictures speak a thousand words
import seaborn as sns
sns.countplot(diabetes["Outcome"], label="Count")
plt.show()

#Lets check the dataframe for any missing values
print(diabetes.isnull().sum())
print(diabetes.describe())
#After observing the dataset we can observe that there are 0s as values in Blood Glucose, BloodPressure, skin thickness,
#Insulin, BMI which is most likely not possible
#Deleting the values could result in loss of valuable information in other predictors
#Imputing mean/average values is not the smartest idea in this case as it is individual readings
#So we conclude that the dataset is not reliable

#We will remove the rows which the �BloodPressure�, �BMI� and �Glucose� are zero.
diabetes_mod = diabetes[(diabetes["BloodPressure"] != 0) & (diabetes["BMI"] != 0) & (diabetes["Glucose"] != 0)]
print("Dimensions of the modified diabetes dataset is {}".format(diabetes_mod.shape))
print(diabetes_mod["BloodPressure"])
#Lets use PCA(Principal Component Analysis) to visualize our data and see if we can achieve good separation
from sklearn.preprocessing import StandardScaler
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']
# Separating out the features
x = diabetes_mod.loc[:, features].values
# Separating out the target
y = diabetes_mod.loc[:,['Outcome']].values


# Standardizing the features
x = StandardScaler().fit_transform(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)

principalDF = pd.DataFrame(data= principalComponents, columns= ['Principal Component 1', "Principal Component 2"])
print("The number of rows in PrincipalDF is {}". format(principalDF.shape))
principalDF.reset_index(drop=True, inplace=True)
diabetes_mod['Outcome'].reset_index(drop=True, inplace=True)
frames = [principalDF, diabetes_mod['Outcome']]
finalDf = pd.concat(frames, axis=1)
print("The number of rows in PrincipalDF Outcome is {}". format(diabetes_mod["Outcome"].shape[0]))

finalDf['Outcome'].replace(0, "Non-Diabetic")
print("The number of rows in finalDF is {}". format(finalDf.shape[0]))
fig= plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)
 
targets = [0, 1]
colors = ['r', 'g']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['Outcome'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'Principal Component 1'],
               finalDf.loc[indicesToKeep, 'Principal Component 2'],
               c = color,
               s = 50
               )
ax.legend(targets)
ax.grid()
plt.show()
plt.savefig('PCA_plot.png')
