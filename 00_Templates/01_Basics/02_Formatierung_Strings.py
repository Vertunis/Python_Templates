name = "Jan"
age = 26
gender = "Male"

# Möglichkeit 1
# %-Formatter
message = "Hello my is is %s i am %i years old and i am a %s person" % (name, age, gender)
print(message)

# Möglichkeit 2
# .format
message = "Hello my is is {} i am {} years old and i am a {} person" % (name, age, gender)
print(message)

# Möglichkeit 3
# f-String
message = f"Hello my is is {name} i am {age} years old and i am a {gender} person"
print(message)