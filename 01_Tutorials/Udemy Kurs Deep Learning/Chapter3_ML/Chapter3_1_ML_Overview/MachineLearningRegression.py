import matplotlib.pyplot as plt

from tf_utils.dummy_data import regression_data # Importiert aus dem Ordner ..\utils unseren DatenGenerator: dummy_data


if __name__ == "__main__":
    x, y = regression_data()

    plt.scatter(x, y)
    plt.show()