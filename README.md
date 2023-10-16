## This was a project to classify scans of breast tissue as either a malignant tumor, benign tumor, or normal tissue by using a Multi-Layer Perceptron

### The data is from ultrasound images among women in ages between 25 and 75 years old. This data was collected in 2018. There are 780 images from 600 patients. The images are in PNG format. The images are categorized into three classes, which are normal, benign, and malignant. 

### This script splits the data into training, vaildation, and test subsets. The script utilitzes PyTorch to load the data in minibathces and creates a simple MLP with 4 hidden layers, including batchnorm and dropout to classify images as either malignant, benign, or normal tissue. It also logs training and validation accuracy after each epoch before evaluation on the test data. 

### The MLP can be run by downloading the data and running "python3 MLP.py" in the command line

## References 
### Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. Data in Brief. 2020 Feb;28:104863. DOI: 10.1016/j.dib.2019.104863.