import pandas as pd

table = pd.read_csv("crops.csv")
feature1 = input("Enter Crop")

Values = []

for x in table[table["Crop_Name"] == feature1]:
    y = table[1:6]
    Values.append(y)

print(Values)