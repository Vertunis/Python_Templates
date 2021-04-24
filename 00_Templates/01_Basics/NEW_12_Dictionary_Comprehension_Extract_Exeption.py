import numpy as np
##############################################################################
# Dictionary in Numpy Array umwandeln
##############################################################################
dict = {"a": 1, "b": 2, "c": 3, "d": 4}
data = list(dict.items())
an_array = np.array(data)
print(an_array)

##############################################################################
#  Extract specific keys with items from dictionary into a New Dictionary
##############################################################################
test_dict ={"Key 1": [1, 2, 3, 4, 5],
            "Key 2": [6, 7, 8, 9, 10],
            "Key 3": [11, 12, 13, 14, 15]}

# Spezifische Einträge aus Dictionarys mit & Operator extrahieren
# https://www.geeksforgeeks.org/python-extract-specific-keys-from-dictionary/
new_dict = {key: test_dict[key] for key in test_dict.keys() & {'Key 1', 'Key 2'}} # Sucht in Keys nach Key 1 und 2 und gibt nur diese aus
print(new_dict)

# Alle Einträge aus Dictionarys außer spezifische Einträge mit "not in" Blacklist_Set übernehmen
# https://stackoverflow.com/questions/8717395/retain-all-entries-except-for-one-key-python
blacklisted_set = ["Key 1", "Key 2"] # Alternativ {"Key 1", "Key 2"}

new_dict_2 = {key: test_dict[key] for key in test_dict.keys() if key not in blacklisted_set} # Sschgreibt alle Keys und Items die nicht in der Blacklist sind in neues Dictionary
print(new_dict_2)


# Alle Einträge aus Dictionarys außer spezifische Einträge mit "in" Whitelist_Set übernehmen
whitelist_Set = ["Key 1", "Key 2"] # Alternativ {"Key 1", "Key 2"}

new_dict_3 = {key: test_dict[key] for key in test_dict.keys() if key in whitelist_Set} # Sschgreibt alle Keys und Items die nicht in der Blacklist sind in neues Dictionary
print(new_dict_3)