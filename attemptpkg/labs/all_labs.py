def show_lab1():
    code = """
#program 1
import pandas as pd
from scipy import stats

df = pd.read_csv("Dataset_1.csv")

# 1. Missing values + handling
print(df.isnull().sum())
df['Occupation'] = df['Occupation'].fillna('Unknown')
df['Satisfaction_Level'] = df['Satisfaction_Level'].fillna(df['Satisfaction_Level'].mean())
df = df.dropna(subset=['Income'])
print("After fill/drop:\n", df.isnull().sum())

# Impact (short note):
# Filling Occupation adds a new category → may affect grouping results.
# Filling Satisfaction_Level changes averages → may raise or lower mean satisfaction.

# 2. Custom binary Satisfaction_Level
df['Sat_Binary'] = df['Satisfaction_Level'].apply(lambda x: 'High' if x>0.7 else 'Low')
print(df['Sat_Binary'].head())

# 3. Map Purchase_History
df['Purchase_Num'] = df['Purchase_History'].map({'High':2,'Medium':1,'Low':0})
print(df['Purchase_Num'].head())

# 4. Outliers (Z-score)
df['Income_Z'] = stats.zscore(df['Income'])
outliers = df[df['Income_Z'].abs()>3]
print("Outliers:\n", outliers[['Income','Income_Z']])


# 5. Missing Years_Employed + fill
df['Years_Employed'] = df['Years_Employed'].fillna(df['Years_Employed'].median())
print("Years_Employed missing after fill:", df['Years_Employed'].isnull().sum())

"""
    print(code)
# This file contains all lab programs with show_labX() printers.

def show_lab2():
    code = """
#program 2
import pandas as pd
df = pd.read_csv("Dataset_2.csv")

# 1. Fill missing Age + City
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['City'] = df['City'].fillna('Unknown')
print("After filling:\n", df[['Age','City']].head())

# 2. Remove duplicates
df = df.drop_duplicates()
print("After removing duplicates:\n", df.head())

# 3. Fix Gender values
df['Gender'] = df['Gender'].replace({'M':'Male','F':'Female'})
print("After fixing Gender:\n", df[['Gender']].head())

# 4. Age ranges
df['Age_Group'] = pd.cut(df['Age'], bins=[18,30,40,50],
                         labels=['18-30','30-40','40-50'])
print("Age Groups:\n", df[['Age','Age_Group']].head())

# 5. City → Dummy variables
city_dummies = pd.get_dummies(df['City'], prefix='City')
print("City Dummies:\n", city_dummies.head())

"""
    print(code)

def show_lab3():
    code = """
#lab 3
import pandas as pd
import numpy as np
sales_df=pd.read_csv('Dataset_3_Sales.csv')
feedback_df=pd.read_csv('Dataset_3_Feedback.csv')
#1
hierarchy = sales_df.set_index(["Product","Month"])
print(hierarchy)       
#2
inner = sales_df.merge(feedback_df, on = "OrderID", how = 'inner')
print(inner)   
#3
q1 = sales_df.copy()
q2 = sales_df.copy()
vertical_concat = pd.concat([q1,q2],axis = 0)
horizontal_concat = pd.concat([q1,q2],axis = 1)
print(vertical_concat)
print(" ")
print(horizontal_concat)
#4
merged = pd.merge(sales_df, feedback_df, on = "OrderID", how = 'outer')
#5
pivoted = sales_df.pivot(index = 'Product', columns = 'Month', values = 'Sales')
print(pivoted)
"""
    print(code)


def show_lab4():
    code = """
# Program 4: Gradient Descent (GD) vs Stochastic Gradient Descent (SGD)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Load + preprocess
X, y = load_iris().data, load_iris().target.reshape(-1, 1)
y = OneHotEncoder(sparse_output=False).fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
sc = StandardScaler()
X_train, X_test = sc.fit_transform(X_train), sc.transform(X_test)

# Model builder
def build(): 
    return Sequential([
        Dense(64, activation='relu', input_shape=(4,)),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')
    ])

# Train GD
gd = build()
gd.compile(optimizer=SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
gd_hist = gd.fit(
    X_train, y_train, 
    epochs=50, batch_size=32, 
    validation_data=(X_test, y_test), 
    verbose=0
)

# Train SGD
sgd = build()
sgd.compile(optimizer=SGD(0.01), loss='categorical_crossentropy', metrics=['accuracy'])
sgd_hist = sgd.fit(
    X_train, y_train, 
    epochs=50, batch_size=1, 
    validation_data=(X_test, y_test), 
    verbose=0
)

# Plotter
def plot(h, key, title):
    plt.plot(h['accuracy'], label='Train '+key)
    plt.plot(h['val_accuracy'], label='Val '+key)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(h['loss'], label='Train '+key)
    plt.plot(h['val_loss'], label='Val '+key)
    plt.title(title + " Loss")
    plt.legend()
    plt.grid()
    plt.show()

plot(gd_hist.history, "GD", "GD Accuracy")
plot(sgd_hist.history, "SGD", "SGD Accuracy")
"""
    print(code)

def show_lab5():
    code = """
#program 5
import pandas as pd
df = pd.read_csv("Dataset_5.csv")

# 1. Total revenue by salesperson per date
print("1:", df.pivot_table(values='revenue', index='salesperson', columns='date', aggfunc='sum'))

# 2. Average revenue per product
print("2:", df.groupby('product')['revenue'].mean())

# 3. Max units sold in one transaction per salesperson
print("3:", df.groupby('salesperson')['units_sold'].max())

# 4. % revenue by region
reg = df.groupby('region')['revenue'].sum()
print("4:", (reg / reg.sum()) * 100)

# 5. Salesperson with most transactions
print("5:", df['salesperson'].value_counts())

# 6. Pivot: total revenue + total units sold by salesperson × product
print("6:", df.pivot_table(values=['revenue','units_sold'],
                           index='salesperson',
                           columns='product',
                           aggfunc='sum'))

# 7. Units sold per region per date
print("7:", df.pivot_table(values='units_sold', index='region', columns='date', aggfunc='sum'))

"""
    print(code)

def show_lab6():
    code = """
import pandas as pd
import matplotlib.pyplot as plt

# Load datasets
g = pd.read_csv("Dataset_6_Games.csv")
p = pd.read_csv("Dataset_6_Players.csv")

# 1. Points change over the season
print("1. Points over season:\n", g[['Game_ID','Team_Points']])
plt.plot(g['Game_ID'], g['Team_Points'])
plt.title("Points Over The Season")
plt.xlabel("Game Number")
plt.ylabel("Points Scored")
plt.show()

# 2. Average attendance
# Removed as 'attendance' column does not exist
print("2. Average attendance:", g['Attendance'].mean())

# 3. Player who scored most + bar chart
player_points = p.groupby('Player')['Points'].sum()
print("3. Top scorer:", player_points.idxmax())
player_points.plot(kind='bar')
plt.title("Total Points by Player")
plt.xlabel("Player")
plt.ylabel("Total Points")
plt.show()

# 4. Games scored above threshold + scoring ranges bar chart
threshold = 100
print("4. Games above threshold:", (g['Team_Points'] > threshold).sum())

bins = [80, 90, 100, 110, 120]
g['range'] = pd.cut(g['Team_Points'], bins)
g['range'].value_counts().sort_index().plot(kind='bar')
plt.title("Number of Games in Score Ranges")
plt.xlabel("Score Range")
plt.ylabel("Number of Games")
plt.show()

# 5. Best performance vs opponents + bar chart
opp_points = g.groupby('Opponent')['Team_Points'].mean()
print("5. Best opponent:", opp_points.idxmax())
opp_points.plot(kind='bar')
plt.title("Average Points vs Opponents")
plt.xlabel("Opponent")
plt.ylabel("Avg Points Scored")
plt.show()

#6. Attendance vs opponents bar chart
print("6. Attendace vs opponents")
g.groupby('Opponent')['Attendance'].mean().plot(kind='bar')
plt.title("Average Attendance vs Opponents")
plt.xlabel("Opponent")
plt.ylabel("Average Attendance")
plt.show()

# 7. Win-Loss record vs points scored (grouped bar chart)
g['Win_Loss_Indicator'] = g.apply(lambda row: 'Win' if row['Team_Points'] > row['Opponent_Points'] else 'Loss', axis=1)
wl_points = g.groupby('Win_Loss_Indicator')['Team_Points'].mean()
wl_points.plot(kind='bar')
plt.title("Average Points: Wins vs Losses")
plt.xlabel("Win / Loss")
plt.ylabel("Avg Points")
plt.show()
"""
    print(code)

def show_lab7():
    code = """
#program 7
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset_7.csv")

df['date'] = pd.to_datetime(df['date'])
df['day']  = df['date'].dt.day_name()

# 1. Heatmap: trips by day of week and base (no hours available)
p1 = df.pivot_table(values='trips', index='day', columns='dispatching_base_number', aggfunc='sum')
sns.heatmap(p1); plt.title("Heatmap: Trips by Day & Base"); plt.show()

# 2. Line chart: trips across a month (example: January)
jan = df[df['date'].dt.month == 1]
jan.groupby(jan['date'].dt.date)['trips'].sum().plot()
plt.title("Trips Trend - January"); plt.ylabel("Trips"); plt.show()

# 3. Bubble chart: total trips per dispatching base
reg = df.groupby('dispatching_base_number')['trips'].sum()

plt.figure(figsize=(10,6))
plt.scatter(reg.index, reg.values, s=reg.values*0.1)
plt.title("Bubble Chart: Trips by Dispatching Base")
plt.xlabel("Dispatching Base")
plt.ylabel("Total Trips")
plt.show()

"""
    print(code)

def show_lab8():
    code = """
#program 8
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset_8.csv")

# 1. Bar chart: survival rate by Pclass
df.groupby('Pclass')['Survived'].mean().plot(kind='bar')
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# 2. Pie chart: survivors vs non-survivors
df['Survived'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Survivors vs Non-Survivors")
plt.ylabel("")
plt.show()

# 3. Stacked bar chart: survivors vs non-survivors by Pclass & Sex
tab = df.groupby(['Pclass','Sex'])['Survived'].value_counts().unstack().fillna(0)
tab.plot(kind='bar', stacked=True)
plt.title("Survival Count by Class & Sex")
plt.ylabel("Count")
plt.show()

"""
    print(code)


def show_lab9():
    code = """
#program 9
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset_9.csv")

# 1. Scatter plot: GrLivArea vs SalePrice
plt.scatter(df['GrLivArea'], df['SalePrice'])
plt.xlabel("GrLivArea"); plt.ylabel("SalePrice")
plt.title("GrLivArea vs SalePrice")
plt.show()

# 2. Heatmap of correlations
num_cols = ['GrLivArea','OverallQual','TotalBsmtSF','SalePrice','YearBuilt']
sns.heatmap(df[num_cols].corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 3. Bubble chart: size = OverallQual

plt.scatter(
    df['GrLivArea'], df['SalePrice'],
    s=df['OverallQual']*20,            # BIG bubbles
    c=df['YearBuilt'], cmap='viridis',   # color = YearBuilt
    alpha=0.5
)
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.title("Bubble Chart: OverallQual(size) & YearBuilt(color)")
plt.colorbar(label="YearBuilt")
plt.xticks(rotation=90)
plt.show()





"""
    print(code)


def show_lab10():
    code = """
#program 10
!pip install squarify
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

df = pd.read_csv("Dataset_10.csv")

# Use only the first listed position
df['MainPos'] = df['player_positions'].str.split(',').str[0]

# 1. Bar chart: number of players per position
df['MainPos'].value_counts().plot(kind='bar')
plt.title("Players per Position"); plt.ylabel("Count")
plt.show()

# 2. Donut chart: rating distribution by position
bins = [0,60,70,80,90,100]
df['RatingRange'] = pd.cut(df['overall'], bins)
sizes = df['RatingRange'].value_counts()
plt.pie(sizes, labels=sizes.index, autopct='%1.1f%%')
plt.title("Rating Distribution")
plt.gca().add_artist(plt.Circle((0,0),0.70,color='white'))
plt.show()

# 3. Treemap (tree diagram): overall rating grouped by position
data = df.groupby('MainPos')['overall'].sum()
squarify.plot(sizes=data.values, label=data.index)
plt.title("Treemap: Total Rating by Position")
plt.axis('off')
plt.show()

# 4. Interpretation
print('
Bar Chart : Shows which positions have the most players.
Donut Chart : Shows how overall ratings are distributed across ranges.
Treemap : Shows the hierarchical structure: which positions contribute most
           to total rating across the dataset.
')
"""
    print(code)


