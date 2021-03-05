import sys
from zipfile import PyZipFile

#Unzip referance: https://docs.python.org/3/library/zipfile.html
for zipLocation in sys.argv[1:]:
    pzf = PyZipFile(zipLocation)
    pzf.extractall()