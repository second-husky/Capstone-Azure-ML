# California Housing Price Prediction by Azure AutoML and Hyperdrive

The goal of this project is to use a California Housing Price dataset to train a machine learning model to predict the housing price based on the given geography, location, income and population information.

Models are built, trained and optimized using both AutoML and Hyperdrive on Microsoft Azure machine learning platform using Azure ML python SDK. The best model will be deployed to web service and can be used to predict housing prices through REST API.

## Project Set Up and Installation

Key steps of this project:
1) Setup the working space and CPU based compute target for the experiments.   
2) Load and clean data from data source and register the dataset to the default datastore
3) Setup an automated ML experiment: run an automated ML experiment to search for and optimize a best model
4) Setup a Hyperdrive experiment and optimize the parameters of a decision tree regression model
5) deploy the best model: compare the results from AutoML and Hyperdrive. AutoML produced the best performing model using Stack Ensemble. The model was deployed as a web service 
6) consume model endpoints: consume the endpoint by sending http request to use the best model to make predictions based on new input data

## Dataset


### Overview

This dataset contains California Housing Price data downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/ and was used as an example in Hands-ON Machine Learning with Scikit-Learn & TensorFLow by Aurelien Geron.The dataset is an adapted version from the original data from the Statlib repository collected from the 1990 California census.

### Task

The dataset will be used as a baseline data to train the machine learning model, which will be used to predict the median housing price within an area. 
The dataset has 10 columns including factors that influecing the housing price. Among them, columns longitude, latitude, housing_mdeian_age, total_rooms, total_bedrooms, population, households and median_income are numerical features. Ocean_proximity is a character feature. Median housing price is the label column and the value to be predicted using the model

### Access

Data was loaded as a pandas dataframe from the URL pointing to a csv file and then registered as a dataset in the datastore.

## Automated ML

In automl settings, "iterations" was set at 20 and experiment_timeout_minutes was set at 20 mins to make sure the experiment finish within expected time frame. "max_concurrent_iterations" was set at 5 considering the computing setting. "primary_metric" was set as r2 score since the model is regression type.

In the configuration of the auto ML run, the CPU based compute target, the registered dataset,the name of the label column, and the above automl settings were passed in. "task" was set to "regression". Early stopping was enabled to save training time. Auto featurization was enabled to explore hidden key combinations of features and to get a more accurate training results. Debug log was enabled for debug purpose. 

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

The best model produced by autoML used stack ensemble algorithm. It achieved a R2 score of 0.84563. 'Ensemble_iterations of the model is 15. Feeding more new data to increase the total sample size may improve the metric further. Or to adjust the weight of individual algorithms in the "stack".

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Decision tree regression was chosen as the training model because it is capable of finding complex nonlinear relationships in the data. Three hyperparameters were tuned using random parameter sampling method. "max_depth" indicates the max of tree depth and the values chosen were 8, 16, 32 adn 64. "Min_samples_split" indicates that the tree will stop branching a node when the number of samples contained in one node is lower than that value. The values chosen for "Min_samples_split" were 50, 100, 500 and 1000. "Min_samples_leaf" indicates that the tree will stop branching a node when the number of samples in its child node is lower than that value. The values chosen for "Min_samples_leaf" is 10, 50 and 100. Tuning "Min_samples_split" and "Min_samples_leaf" to avoid overfitting, which is a common issue with decision tree models

The early stopping policy with a slack factor of 0.1 and an evaluation interval of 1 according to low max_toal_runs and max_concurrent_runs settings. Any run that shows accuracy below 10% of the best-performing run will be terminated and this policy will evaluated on every run. This will help save time of executing the low-performance runs

### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The best decision tree regression model optimized by hyperparameter tuning is 0.759. The best parameter combination within the sampling range is "Max Depth" = 32, "min samples split" = 50 and "min samples leaf" = 10. The model can be further improved by expand the sampling range of hyperparameteres and monitor the R2 score dependency on them. Or try other models like XGBoost.

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

Inference configuration was created by taking the deploy environment settings and the entry script downloaded from the best run produced by autoML. The registered model was deployed as a web service endpoint on an Azure Container Instance. To query the prediction using the model, a http request was sent to the REST endpoint with the input data including values for all the features as a json format string. The predicted median housing price was retrieved and printed as the output 

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response
