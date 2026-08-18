### INSTRUCTIONS TO RUN TRAINING FOR DATASET-1 (MNIST)

Compile the backend_dataset_1.cpp as follows:

g++ -O3 -shared -std=c++17 -fPIC     backend_dataset_1.cpp     -o custom_core_dataset_1$(python3-config --extension-suffix)     `python3 -m pybind11 --includes`     `pkg-config --cflags --libs opencv4`

Run python3 train_dataset_1.py 

It will ask for path to the dataset :  ./data/data_1/data_1

Run python3 eval_dataset_1.py

It will ask for path to the dataset :  ./data/data_1/data_1

It will aslo ask for the saved model weights :     ./checkpoints/model_saved.pkl


### INSTRUCTIONS TO RUN TRAINING FOR DATASET-2 (CIFAR-100)

Compile the backend_dataset_2.cpp as follows:

g++ -O3 -shared -std=c++17 -fPIC     backend_dataset_2.cpp     -o custom_core_dataset_2$(python3-config --extension-suffix)     `python3 -m pybind11 --includes`     `pkg-config --cflags --libs opencv4`

Run python3 train_dataset_2.py  

It will ask for path to the dataset :  ./data/data_2/data_2

Run python3 eval_dataset_2.py

It will ask for path to the dataset :  ./data/data_2/data_2

It will aslo ask for the saved model weights :     ./checkpoints/model_dataset_2.pkl

To get plots for dataset 2 

Run python3 plot_dataset_2.py

Please put the datasets in the main project directory which is the same directory as this readme file

