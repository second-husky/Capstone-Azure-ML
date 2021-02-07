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
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
