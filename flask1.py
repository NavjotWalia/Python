from flask import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

table = pd.read_csv("crop_details.csv")

# Creating a Python App running on Flask Server
app = Flask(__name__)

@app.route("/")
def about():
    return render_template("about.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/revert-predict', methods=['GET', 'POST'])
def revertPredict():
    return render_template("home.html")

@app.route('/predict-label', methods=['GET', 'POST'])
def predictLabel():
    feature1 = request.form['feature1']     #State
    feature2 = request.form['feature2']     #Crop
    feature3 = request.form['feature3']     #Year

    df = pd.read_csv("crop_details.csv")

    X = []
    Y = []

    States = feature1
    Crop = feature2

    for x in df[(df["STATES"] == States) & (df["Crop"] == Crop)]["YEAR"].values:
        X.append(x)

    for y in df[(df["STATES"] == States) & (df["Crop"] == Crop)]["Production"].values:
        Y.append(y)

    X = np.reshape(X, (-1, 1))
    Y = np.reshape(Y, (-1, 1))

    scalerFeatures = MinMaxScaler()
    scalerTarget = MinMaxScaler()

    scalerFeatures.fit(X)
    scalerTarget.fit(Y)

    XScale = scalerFeatures.transform(X)
    YScale = scalerTarget.transform(Y)

    # Preparing Training and Testing Data
    X_train, X_test, Y_train, Y_test = train_test_split(XScale, YScale)

    # Step2: Creating ANN Model for Regression
    model = keras.Sequential()
    model.add(keras.layers.Dense(12, input_dim=1, activation='relu'))  # Input Layer
    model.add(keras.layers.Dense(8, activation='relu'))  # Hidden Layer
    model.add(keras.layers.Dense(1, activation='linear'))  # Hidden Layer

    # Compile the Model
    model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

    # Training the Model -> We will get a history object
    history = model.fit(XScale, YScale, epochs=25, batch_size=60, validation_split=0.2)

    XToBePredicted = np.array([[feature3]])
    XToBePredictedTransformed = scalerFeatures.transform(XToBePredicted)
    YPredicted = model.predict(XToBePredictedTransformed)  # it will be in MinMaxScaler i.e. range between 0 to 1

    # Invert Normalization i.e. get the real value again
    YPredicted = scalerTarget.inverse_transform(YPredicted)
    print(">> FOR X:{} PREDICTION Y:{}".format(XToBePredicted, YPredicted))

    print(history.history.keys())
    output = YPredicted
    return render_template("outputprediction.html", output=output)

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/graphhome")
def graphhome():
    return render_template("graphhome.html")

@app.route("/graphtp")
def graphTp():
    return render_template("graphtp.html")

@app.route('/graph-label', methods=['GET', 'POST'])
def graphLabel():

    table = pd.read_csv("crop_details.csv")
    feature1 = request.form['feature1']     #State
    feature2 = request.form['feature2']     #crop

    ProductionValues = []
    YearValues = []

    for x in table[(table["STATES"] == feature1) & (table["Crop"] == feature2)]["YEAR"].values:
        YearValues.append(x)

    for y in table[(table["STATES"] == feature1) & (table["Crop"] == feature2)]["Production"].values:
        ProductionValues.append(y)

    plt.plot(YearValues,ProductionValues)
    plt.ylabel("Total Production")
    plt.xlabel("Years")
    plt.legend("Total Production of {}".format(feature2))
    plt.show()
    plt.savefig('C:/Users/dell/PycharmProjects/CropProduction/static/image.jpg', bbox_inches='tight')

    return render_template("home.html")

@app.route("/exports")
def exports():
    return render_template("exports.html")

@app.route('/export-label', methods=['GET', 'POST'])
def exportLabel():

    table = pd.read_csv("crop_details.csv")
    feature1 = request.form['feature1']     #State
    feature2 = request.form['feature2']     #crop

    ExportValues = []
    YearValues = []

    for x in table[(table["STATES"] == feature1) & (table["Crop"] == feature2)]["YEAR"].values:
        YearValues.append(x)

    for y in table[(table["STATES"] == feature1) & (table["Crop"] == feature2)]["Export"].values:
        ExportValues.append(y)

    plt.plot(YearValues, ExportValues)
    plt.ylabel("Exports")
    plt.xlabel("Years")
    plt.legend("Exports of {}".format(feature2))
    plt.show()
    plt.savefig('C:/Users/dell/PycharmProjects/CropProduction/static/image.jpg', bbox_inches='tight')

    return render_template("home.html")


if __name__ == '__main__':
    app.run(debug=True)