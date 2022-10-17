import os 
import sys
sys.path.insert(0, os.path.abspath('../../'))

from nnlearn.util import ScriptInformation

def your_test():
    s = ScriptInformation()
    s.hello()

if __name__ == '__main__':
    your_test()

