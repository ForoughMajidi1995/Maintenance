Hello! This repository is the implementation of the paper "An Efficient Model Maintenance Approach for MLOps". SimReuse notebook help you to 1) store you previously trained models, 2) reuse them if your test data distribution in the production environment is similar to the one of the previous training data distributions. 

* To run the SimReuse notebook you need to provide the following items as input: ) dataset (a pandas dataframe), 2) window_size (an integer number), 3) target_column (name of your target column as a string), 4) windows_similarity_dictionary (a dictionary that its keys are the index of preceding data segments with similar data distribution to the data segment distribution with index of value ), forecasting_approach (a string that identifies the forecasting method you want to use)

* The GRU folder contains the implementations and the results of training GRU models and testing them on five different datasets.
* The lstm folder contains the implementations and the results of training lstm models and testing them on five different datasets.
* The RF folder contains the implementations and the results of training Random Forest models and testing them on five different datasets.
* The XGB folder contains the implementations and the results of training XGB models and testing them on five different datasets.
* Variable_to_specify_name_of_the_dataset files include the variables related to each dataset, like target column name, start_date, end_date, and so on. You need to make a similar .py file for your own dataset.
* training_utilities file includes the utilities to preprocess and train the models.
* training_utilities_2nd_part file includes the utilities to apply ACF function, find seasonalities, training LSTM and GRU models and so on.


This is the link to the paper: https://arxiv.org/abs/2412.04657
