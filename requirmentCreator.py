import os
import re
from os import system as cmd
# from glob import glob as search

class RequirmentCreator:
    def __init__(self):
        pass
    
    def __getAllReqirments(self):
        requirments = {}
        
        cmd("pip freeze > requirments.txt")
        
        with open("requirments.txt", "r") as f:
            read = f.read()
        
        for lib in read.split("\n"):
            if lib.count("==") == 1:
                lib, version = lib.split("==")
                requirments[lib] = version

        return requirments
    
    def __removeRedandentRequirments(self, requirmentsMap, currentDirectory):
        realReqirments = []
        allFilesRequirments = set()
        allFiles = self.__getAllFiles(currentDirectory)
        
        for fileName in allFiles:
            allFilesRequirments.update(self.__getRequirmentFromFile(fileName))

        for lib in allFilesRequirments:
            if requirmentsMap.get(lib) is not None:
                realReqirments.append("{}=={}".format(lib, requirmentsMap[lib]))
        
        return realReqirments
    
    def __getRequirmentFromFile(self, fileName):
        allRequirments = set()
        
        with open(fileName, 'r') as f:
                read = f.read()

        if read.count("import ") == 0 and read.count("from") == 0:
            pass
        else:
            lines = read.split("\n")
            for line in lines:
                if "#" in line:
                    continue
                
                m = re.search(r'(?<=from )\w+', line)
                if m is not None:
                    allRequirments.add(m.group())
                else:
                    m = re.search(r'(?<=import )\w+', line)
                
                    if m is not None:
                        allRequirments.add(m.group())
        
        return allRequirments
    
    @staticmethod
    def __getAllFiles(currentDirectory):
        allFiles = []
        for dirpath, dirs, files in os.walk(currentDirectory):
            if files != ".git":
                for filename in files:
                    fname = os.path.join(dirpath, filename)
                    if fname.endswith(".py"):
                        allFiles.append(fname)
        return allFiles
    
    def __writeToFile(self, data, currentDirectory = None, fileName = "requirments.txt"):
        with open("{}\{}".format(currentDirectory, fileName), 'w') as f:
            f.write(data)
    
    def createRequirmentsFile(self, searchDirectory = None):
        searchDirectory = os.getcwd() if searchDirectory is None else searchDirectory
        requirmentsMap = self.__getAllReqirments()
        realReqirments = self.__removeRedandentRequirments(requirmentsMap, searchDirectory)
        self.__writeToFile("\n".join(realReqirments), searchDirectory)

if __name__ == '__main__':
    print(RequirmentCreator().createRequirmentsFile("C:\\Users\\Tamir\\Downloads\\computer vision\\CV-LaneDetection"))
