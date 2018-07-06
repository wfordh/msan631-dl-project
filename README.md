# msan631-dl-project
Deep Learning/Computer Vision webcam digit recognizer project

Course project for MSAN631 Deep Learning with PyTorch taught by Yannet Interian. Inspiration for this project was Akshay Bahadur's Digit Recognizer project, which can be found here: 
- https://github.com/akshaybahadur21/Digit-Recognizer

The base structure of the model and the webcam operations were taken from his repository, and then Taylor Pellerin (@tjpell) and I translated the model frameworks from NumPy to PyTorch. The models were trained on the standard MNIST dataset before being put into production by linking them with the webcam.

TO DO:
- Clean up code and get consistent syntax. 
- Put the neural net model into a py file.
- Improve webcam connection. Currently the predictions when implemented with the webcam are not very good, despite good test results for the models. 
- Get finger pen to work so user can draw digits with their fingers instead of having to write them on paper before testing.  
