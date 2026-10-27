# O Level Students’ Mathematics Score Prediction

## Project Description
The objective of this project is to predict Students’ O Level Mathematics Exam Score. This prediction would help to identify the weaker students prior to the examination. To achieve this three Machine Learning Models will be developed, from which one best model will be selected to achieve the object.

---

## Prerequisites and Installations:

### Preresuites:

- *Anaconda or Miniconda - powerful distribution of Python.*
- *Git - Version control system that allows to track changes in code.*
- *Visual Studio Code or VS Code - A lightweight but versatile code editor packed with features that make coding more efficient.*

### Installations:

Open Anaconda Prompt and create a conda environment - schenv for this project, then activate it. Install the following packages, which could be found in requirements.txt

1. numpy
2. matplotlib
3. pandas
4. pip
5. PyYAML
6. scikit-learn
7. seaborn

Open VS Code, select Terminal from View menu, in the below horizontal panel use dropdown menu just near the + sign and select Miniconda, then activate the schenv environment. Then, use the Extensions icon from the left panel and select and install both python and jupyter notebook. Use 'Show and Run Commands (Control + Shift + P)' to select the Python Create Environment to create the environment schenv and open the jupyter notebook and select the kernel - schenv(python 3.11.17) from the list. Now, you can start to use the Jupyter Notebook to do EDA, Data Cleaning and Preprocessing, etc.

## Overview of Key Findings of EDA:  

    Import the Pandas. With the help of pandas load CSV data from durectiry where it is downloaded. Then start to explore the data. The followings are the findings of EDA.
 
  1. There are 15,900 rows and 18 Columns (including Index column)

  2. The data types are float64(6), int64(2), str(10) - value in the bracket is the no. of columns

  3. The following columns are having either empty cells or 'None' value
     - 'CCA' - has 3829 cells with value 'None' and it will be considered as empty cells
     - 'ÇCA' - has 130 cells with value 'NONE', so the total no. of cells with 'None' or 'NONE' is 3959
     - 'final_test' - has 495 empty cells
     - 'attendance_rate' - has 798 empty cells

  4. Some of the cells of column 'CCA' has values 'Sports', 'Arts' and 'Clubs' and some with fully caps.

  5. The 'tuition' column values should be either 'Yes' or 'No' but 588 cells have values 'Y' or 'N'

  6. The 'age'column values should be either 15 or 16, but 404 cells have values -4, -5, 5, and 6

  7. The columns 'sleep_time' and 'wake_time' are in string format.

  8. There are 763 duplicated rows and it is detucted based 'student_id' on the assumption that there should not be repetition of details of same student.   

## Feature Handling Description:

  1. The rows, where the values of 'final_test' and 'attendance_rate' are empty, were dropped, since the value cannot be filled based on a valid assumption.

  2. 3959 rows, where the value of the column 'CCA' are either 'None' or 'NONE', both should be taken as a seperate categorical value of the coulmn 'CCA'and should not be dropped. It should be replaced by the value 'No_CCA'.

  3. 588 rows, where the values of 'tuition' column are either 'Y' or 'N', these should be replaced by the values 'Yes' for 'Y' and 'No' for 'N'.

  4. 404 rows, where the values of 'age' are either -4, -5, 5 and 6. On the assumption that wrong values were entered dur to typo error. so, it is decided to convert -5, 5 into 15, and 6 into 16. Moreover, the minimum age of the O Level students should be 15. So, -4 cannot be converted into 14 and so that row will be dropped.

  5. 763 duplicated rows due to repetition of 'student_id', will be dropped.

  6. Both 'sleep_time' and 'wake_time' columns are in string format. As such both column cannot used so this should be convereted into datetime format. Then, we can calculate the sleep duration and it can be saved under the name 'duration_hours' and it can be used for the prediction.

  7. Both 'n_male' and 'n_female' features cannot be used individually for the prediction, instead it can be summed up to use for the prediction and it shuld be saved under the name 'total_class_size'.

  8. Now, the total number features will become 20, of which the following features won't be used in prediction of O Level mathematics 'final_test', as they are noway directly linked with the scores in mathematics, 'student_id', 'mode_of_transport', 'bag_color', 'sleep_time', 'wake_time', 'n_male', 'n_female', 'index'. So, these features will be dropped.

  9. Columns with binary string values, 'gender', 'tuition' and 'direct_admission' should be converted into 0, 1.

## Instruction for Executing the Pipeline:

On execution of python script 'main.py', with the help of config-yaml file, it will set the basic parameters - to load data, splitting ratio of data, param_grid, numerical features, nominal_features etc., and it will help to execute the pipeline, which will do data preparation, data cleaning and preprocessing, data splitting, modelling, hyperparameter tuning, metrics settings, training the model with the help of train data, validation with the help of validation set, selection of best model and testing the best model on test data.

## Logical Steps/Flow of Pipeline:

The main.py script compiles and orchestrates the entire pipeline by:

- Loading and validating configuration and data

- Initializing data preparation and model training

- Running the training and evaluation process

- Train and evaluate tuned models with hyperparameter tuning

- Combine all models and their metrics into dictionaries

- Finding the best model based on R² score

- Identifying and evaluating the best model on the test set

- Predicting scores for new students with a saved model

## Explanation of Model Choices

### Implemented Models

1. \*\*Linear Regression\*\*: Basic model without regularization because it serves as a simple, interpretable baseline against which the performance of the other models can be measured.

2. \*\*Ridge Regression\*\*: Linear regression with L2 regularization because it shrinks coefficient magnitudes to reduce overfitting and stabilize estimates without eliminating any features entirely, which is useful given the correlation between `attendance\_rate` and `duration\_hours`.

3. \*\*Lasso Regression\*\*: Linear regression with L1 regularization because it can shrink some coefficients exactly to zero, effectively performing feature selection, which helps identify whether any of the weaker predictors add little to the model.

### Hyperparameter Tuning

- Grid search/Randomized search is performed for Ridge and Lasso models because each has one main hyperparameter, `alpha`, and both search systematically evaluates every value from 0.1 to 1000 on a log scale, so the best value is not stuck at the edge of the search range.

- `fit_intercept` is not tuned and is left at `True` because the intercept is the model's starting prediction for an average student, which should be close to the average score of about 67. Setting it to `False` would force this starting prediction to zero.

- The final model is selected using cross-validation R² because it shows both typical performance and how stable it is across folds.

## Evaluation Models

### Eavluation Metrics

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual scores
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual scores
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target
- **R² (Coefficient of Determination)**: Proportion of variance in scores explained by the model

### Model Comparison and Results
The data is split into 80% training (11,107 students), 10% validation (1,388) and 10% test (1,388). All models are compared on the validation set, and the selected model is then evaluated once on the test set.

### Evaluation Results

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Linear Regression | 7.3034 | 83.1257 | 9.1173 | 0.5904 |
| Ridge Regression  | 7.3034 | 83.1261 | 9.1174 | 0.5904 |
| Lasso Regression  | 7.9790 | 97.3301 | 9.8656 | 0.5204 |

## Consideration for Deployment

### Model Comparison based on GridSearch CV

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Ridge Regression  | 7.3040 | 83.1297 | 9.1175 | 0.5904 |
| Lasso Regression  | 7.3138 | 83.3289 | 9.1285 | 0.5894 |

### Model Comparison based on RandomisedSearch CV

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Ridge Regression  | 7.3040 | 83.1297 | 9.1175 | 0.5904 |
| Lasso Regression  | 7.3138 | 83.3289 | 9.1285 | 0.5894 | 

While comparing both the model, the Ridge regression model RMSE value is lesser than that of Lasso model and R2 Value is greater than that of Lasso, so the ridge model is selected as the best model.

### Application of best model on Test Data

Best Ridge Regression Model, Final Test Metrics:

| Metrics | Value |
|---|---|
| Final Test MAE | 7.2248776989153445 |
| Final Test MSE | 81.58832835031212 |
| Final Test RMSE | 9.032625772736969 | 
| Final Test R² | 0.5811176984695714 |


