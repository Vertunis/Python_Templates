# https://stackoverflow.com/questions/38412495/difference-between-os-path-dirnameos-path-abspath-file-and-os-path-dirnam
import os

File = os.path.abspath(__file__)  # Ermittelt kompletten Pfad zu aktuellem File mit Filename und Endung
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # os.path.dirname() function simply removes the last segment of a path.
BASE_DIR = os.path.dirname(PROJECT_ROOT)  # Parent Directory of PROJECT_ROOT

print(File)
print(PROJECT_ROOT)
print(BASE_DIR)
