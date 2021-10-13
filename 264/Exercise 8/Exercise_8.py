from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd

def main():
    shelter_1 = pd.read_csv("shelter_1.csv")
    shelter_2 = pd.read_csv("shelter_2.csv")
    shelter_3 = pd.read_csv("shelter_3.csv")
    shelter_x = pd.concat([shelter_1, shelter_2, shelter_3])

    print(shelter_1.describe())

    print(shelter_x.describe())

    dogs = shelter_x[shelter_x["Animal"] == "Dog"]

    iris_data = datasets.load_iris()
    print(list(iris_data.target_names))
    data = pd.DataFrame(iris_data.data, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
    print(data)
    data.merge(pd.DataFrame(iris_data.target, columns=["Setosa", "Versicolor", "Virginica"]))
    print(data)

main()