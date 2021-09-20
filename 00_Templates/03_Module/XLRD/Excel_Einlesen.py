import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd


# Methoden

# Read Excel data
def read_excel_file(filename):
    book = xlrd.open_workbook(filename, encoding_override = "utf-8")
    sheet = book.sheet_by_index(0)

    tag = u"Anzahl"  # or u"Anz" or ...
    column_0_values = sheet.col_values(colx=0) # Nimmt alle Werte der Spalte 0
    row_0_values = sheet.row_values(rowx=0)

    print(f"Anzahl Zeilen: {len(column_0_values)}")
    print(f"Anzahl Spalten: {len(row_0_values)}")
    for rowx in range(len(column_0_values)-1, 0, -1):

        #print(colx)
        if column_0_values[rowx] == tag:
            lastrow = rowx
            print(f"Letzte Zeile: {lastrow}")
            break

    y_data = []
    flag = 0
    for colx in range(1, len(row_0_values)):
        print(colx)
        print(sheet.cell(3, colx).value)

        y_1 = sheet.cell(4, colx).value
        y_2 = sheet.cell(3, colx).value
        if y_2 == "" and flag == 0:
            y_2 = sheet.cell(3, colx-1).value
            colx_count = colx-1
            print("y_2 isemty")
            flag = 1
        elif y_2 == "" and flag == 1:
            y_2 = sheet.cell(3, colx_count).value

        y_data.append(y_1 + ", " + y_2)

        print(f"Ausgelesene Daten: {y_data}")

    x_data = np.asarray([sheet.cell(lastrow, i).value for i in range(1, len(row_0_values))])

    return x_data, y_data


def plot_bar(y,x):

    std = np.std(x)
    plt.bar(y, x)
    plt.title("DOODLE Grilloptionen")
    plt.ylim(0, 16)
    plt.ylabel("Zusagen")

    #plt.legend(f"Test", loc= "upper left")
    plt.show()


if __name__ == '__main__':

    filename = "Doodle.xls"

    #df = pd.read_excel(filename, sheetname="Umfrage")
    x_data, y_data = read_excel_file(filename)

    plot_bar(y_data, x_data)