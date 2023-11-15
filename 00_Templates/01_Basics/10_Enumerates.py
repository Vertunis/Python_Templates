# Enumerates, einfaches Indexieren und Iterieren von Daten
# https://book.pythontips.com/en/latest/enumerate.html
# Python's enumerate method allows you to iterate over a list, tuple,
# or dictionary and return a tuple containing the index of each element and the element itself.
# This is a convenient way to keep track of both the element and its position on a list.

smaller = [1,2,3]
names = ['jan','peter','horst']

# Automatisch Liste erstellen von Names (beginnt bei 0)
for idx, name in enumerate(names, 0): # Beginnt bei 0
    print(idx, name)

print("\n")

# Macht das gleiche wie oben, nur wird zusätzlich die ZIP Funktion vorher ausgeführt.
# -> Es wird eine Liste aus Smaller und Names erstellt dann zusätzlich iteriert
# Ausgabe: Nummer, Index, Name
for number, (index_names,name) in zip(smaller, enumerate(names, 0)):
    print(number,index_names, name)

print("\n")

# Beispiel mit enumerate und zip anders herum
# Ausgabe: Index Zuerst, dann Nummer, dann Name
for idx, (index_names,name) in enumerate(zip(smaller, names), 0):
    print(idx,index_names, name)