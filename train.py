import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pickle

# Load the coffee dataset
df_coffee= pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")
df_coffee.head()
# Define the features and target variable
X= df_coffee[["100g_USD"]]
y= df_coffee["rating"]
# Create and train the linear regression model
model= LinearRegression()
model.fit(X,y)

# Save the trained linear regression model
with open("model_1.pickle", "wb") as f:
    pickle.dump(model,f)



# Map roast categories to numeric values
roasts = df_coffee["roast"].dropna().unique()
list(enumerate(roasts))
roast_map = {roast: number for number, roast in enumerate(roasts)} 

def roast_category(roast):
    return roast_map.get(roast,-1)
df_coffee["roast_cat"]=df_coffee["roast"].apply(roast_category)

# Define the features and target variable
X =df_coffee[["100g_USD","roast_cat"]]
y= df_coffee["rating"]

# Create and train the decision tree model
model = DecisionTreeRegressor()
model.fit(X, y)

# Save the trained decision tree model
with open("model_2.pickle", "wb") as f:
    pickle.dump(model, f)