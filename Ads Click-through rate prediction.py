import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
pio.templates.default = "plotly_white"

data = pd.read_csv("ad_10000records.csv")
print(data.head())

data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No", 
                               1: "Yes"})

fig = px.box(data, 
             x="Daily Time Spent on Site",  
             color="Clicked on Ad", 
             title="Click Through Rate based Time Spent on Site", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

fig = px.box(data, 
             x="Daily Internet Usage",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Daily Internet Usage", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

fig = px.box(data, 
             x="Age",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Age", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

fig = px.box(data, 
             x="Area Income",  
             color="Clicked on Ad", 
             title="Click Through Rate based on Income", 
             color_discrete_map={'Yes':'blue',
                                 'No':'red'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

data["Clicked on Ad"].value_counts()

click_through_rate = 4917 / 10000 * 100
print(click_through_rate)

data["Gender"] = data["Gender"].map({"Male": 1, 
                               "Female": 0})

x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
y=data.iloc[:,9]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,
                                           test_size=0.2,
                                           random_state=4)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)

from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,y_pred))

print("Ads Click Through Rate Prediction : ")
a = float(input("Daily Time Spent on Site: "))
b = float(input("Age: "))
c = float(input("Area Income: "))
d = float(input("Daily Internet Usage: "))
e = input("Gender (Male = 1, Female = 0) : ")

features = np.array([[a, b, c, d, e]])
print("Will the user click on ad = ", model.predict(features))

