import glob

files = glob.glob("./sample_files/*.txt") # Listet alle .txt Files auf die und dieser Dir sind

print(files)

for file in files:
    with open(file, "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
