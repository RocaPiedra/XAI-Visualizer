import os
print(os.path.join(os.path.dirname(__file__), '..', 'data', 'mydata.json'))
print(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'mydata.json')))
print(os.path.dirname(__file__))