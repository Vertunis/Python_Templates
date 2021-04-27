# https://www.programiz.com/python-programming/nested-dictionary

people = {1: {'name': 'John', 'age': '27', 'sex': 'Male'},      # Dictionary 1
          2: {'name': 'Marie', 'age': '22', 'sex': 'Female'}}   # Dictionary 2

print(people)

# Acces Nested Dictionarys
print(people[1]['name']) # Zugriff Dic 1
print(people[2]['age']) # Zugriff Dic 2
print(people[1]['sex'])