import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import geopandas as gpd
from geopandas.tools import sjoin
import rtree
import seaborn as sns
from scipy import optimize
from math import *
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D

###Read the data
data = pd.read_csv("./Meteorite_Landings.csv")

### Pre-process the data

data = data.dropna()

data["year"] = data["year"].str[6:10]
data["year"] = data["year"].astype(int)
data = data[data["year"] <= 2021]
data = data[data["year"] >= 1975]

data_q1a = data[data['mass (g)'] >= 0.0]
data_q1b = data[data['mass (g)'] <= 50000.0]



#Histogram mass distribution (All)
plt.figure("Histogram mass distribution (All)")
ax = data_q1a["mass (g)"].plot(kind="hist", bins=500,color='brown')
ax.set_yscale('log')

#Histogram mass distribution (Less than 50 000 grams)
plt.figure("Histogram mass distribution (Less than 50 000 grams)")
ax = data_q1b["mass (g)"].plot(kind="hist", bins=50,color='brown')
ax.set_yscale('log')

#Density
# plt.figure("Density")
# axis = data_q1b["mass (g)"].plot.kde()



### Plot of occurences function of years
plt.figure("Occurences function of years")
counter = data.groupby(['year'])['year'].count().reset_index(name="count")
ax = plt.scatter(counter["year"], counter["count"],color='brown')

plt.figure("Occurences function of years with linear regression")
counter = data.groupby(['year'])['year'].count().reset_index(name="count")
ax = plt.scatter(counter["year"], counter["count"],color='brown')


### Linear regression
x = counter.iloc[:, 0:1].values
y = counter.iloc[:, 1].values

regressor = LinearRegression()
regressor.fit(x, y)

y_pred = regressor.predict(x) 

plt.plot(x, y_pred, color='green')


### Prediction

year_to_predict = 2022

nb_prediction = year_to_predict * regressor.coef_ + regressor.intercept_

print(f"Number of Meteorite prediction for {year_to_predict} is: {nb_prediction}\n")

#### Focusing on Oman

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
Oman = world.loc[world['name'] == 'Oman'] # get Oman row
boundaries = Oman['geometry'] # get Oman geometry
type(Oman)
gpd.geodataframe.GeoDataFrame
Oman.head()

## Getting the coords of meteorites 
tmp=[]
tmp.append([])
tmp.append([])
for coord in data["GeoLocation"]:
    partial=coord.split(",")
    x=partial[0]
    x=x[1:]
    tmp[0].append(x)
    y=partial[1]
    y=y[1:-1]
    tmp[1].append(y)

points = gpd.GeoDataFrame(data, geometry = gpd.points_from_xy(tmp[1],tmp[0])) 

## Taking the intersection of Oman and Meteor. coords.
pointInPolys = sjoin(points, Oman, how='inner')

fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_aspect('equal')
Oman.plot(ax=ax, color='gray', edgecolor='black')
ax.set_title('Oman\'s meteorites map')
pointInPolys.plot(ax=ax, color='brown' ) 


## X density
X=pointInPolys['geometry'].x

fig, ax1 = plt.subplots()
sns.kdeplot(data=X, ax=ax1)
ax1.set_xlim((X.min(), 55.2))
ax2 = ax1.twinx()
sns.histplot(data=X, discrete=True, ax=ax2, color='brown' )

## Y density
Y=pointInPolys['geometry'].y

fig, ax1 = plt.subplots()
sns.kdeplot(data=Y, ax=ax1)
ax1.set_xlim((Y.min(), 20.2))
ax2 = ax1.twinx()
sns.histplot(data=Y, discrete=True, ax=ax2, color='brown' )


DataGeo = np.transpose([X,Y])











centre_x =54.58
centre_y =19.14
A=sqrt(953*953+1358*1358)
ecart_x = (54.58-53.52)/0.64
ecart_y = (19.14-17.86)/2
print(A,ecart_x,ecart_y)

#Parameters to set

mu_x = 54.58

variance_x = 0.3



mu_y = 19.14

variance_y = 0.64



#Create grid and multivariate normal

x = np.linspace(53,56,500)

y = np.linspace(17,21,500)

X, Y = np.meshgrid(x,y)

pos = np.empty(X.shape + (2,))

pos[:, :, 0] = X; pos[:, :, 1] = Y

rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])



#Make a 3D plot

fig = plt.figure()

ax = fig.gca(projection='3d')

ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)

ax.set_xlabel('X axis')

ax.set_ylabel('Y axis')

ax.set_zlabel('Z axis')
import math 

def getIndexPoint(rv, a, b, R):
  K = 499
  Sum = 0
  CountSum = 0
  for i in range(K):
    for j in range(K):
      if (math.sqrt(((pos[:, :, 0][i][j])-a)**2+((pos[:, :, 1][i][j])-b)**2))<R:
        Sum += rv.pdf(pos[i][j])
        CountSum +=1
  return (Sum/CountSum)

a = getIndexPoint(rv, 53.9555, 18.9644, 1)
print(a)

# print(f"Pour la question 1a on a :\n{data_q1a.shape}\n")
# print(f"Pour la question 1b on a :\n{data_q1b.shape}\n")
# print(data.dtypes)

# print(data.shape)
plt.show()