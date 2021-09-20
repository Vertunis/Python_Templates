# https://www.touchdown-mathe.de/titel/exponentialfunktion-parameter-bestimmen/
# Die Exponentialfunktion hat die allgmeine Funktionsgleichung a^(ax2+x)+b
#
# Typischerweise sind bei Exponentialfunktionen zwei Arten von Parametern zu bestimmen:
#
#     Parametern innerhalb des Exponenten
#     Parametern außerhalb der Potenz
import numpy as np
import matplotlib.pyplot as plt


def main():

    # Exponentielle Regression
    # y_exponentiell = c*e^(ax**2)+b
    t_end = 564169.0086665154
    x = np.linspace(0, t_end, 100)
    b = 101  # Aus 100%=e**0+b an Stelle t=0
    d = -1  # Angenommen
    a = (np.log((0 - b) * d)) / t_end ** 2

    y = d * np.exp(a * x[:]**2) + b
    y_2 = -1.0001 * np.exp(a * x[:] ** 2) + b

    a_3 = (np.log((0 - b) * d)) / (t_end ** 2 + t_end)
    y_3 = d * np.exp(a_3 * x[:]**2) + b

    plt.figure(0)
    plt.plot(x, y,color='grey')
    plt.plot(x, y_2, color='red')
    plt.plot(x, y_3, color='blue')
    plt.xlabel('$x$')
    plt.ylabel('$\exp(x)$')


    plt.show()

if __name__ == '__main__':
    main()