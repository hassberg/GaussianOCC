# GaussianOneClassClassifier

## Experiment Run
File "_experiment_pipeline/experiment_pipeline_runner.py" defines the entry point for running experiments. 
To execute, run 
```console 
python .\experiment_pipeline_runner.py
```
It requires python3.X and several dependencies. They are listed, if trying to run without being installed. 

Specifically:
- https://github.com/bela127/active_learning_ts.git
- pandas
- torch
- gpytorch
- matplotlib
- ipykernel
- cvxopt
- tqdm
- ...


The Experiment runs for each defined SurrogateModel, with their SelectionCriteria, a learning process on the in "_experiment_pipeline/data_sets" listed data samples. The pipeline process recreates a Map of experiment to run, defined by:
- Surrogate Model
- Selection Criteria
- Data Set
- learning_steps
- experiment_repeats
- best_k_to_run

Later 3 are defined at the beginning of the document. 

best_k_to_run: Best k grid search results to run\
experiment_repeats: repeats on each data sample\
learning_steps: Steps in AL cycle

### Data Samples
Data samples are organized in the following structure:
- data_sets
  - discrete
    - "data set name"
      - grid
        - train.csv
        - test.csv
      - sample 1
        - train.csv
        - test.csv
      - sample ..
        - ..
      - sample n
        - train.csv
        - test.csv
    - "data set"
      - ...
  - continuous
    - "data set name"
      - grid
        - train.csv
        - test.csv
      - ground_truth
        - ground_truth.csv
      - sample 1
        - train.csv
        - test.csv
      - ...
    - ...

This structure has to be kept. In CSV files, the last column contains the class label. Files in "grid" are used for gridsearch, other samples are used to execute an experiment. \
The folders "discrete" and "continuous" are used to distinguish between QueryPoolModes. 

We strongly recommend using the data-sampler
```TODO link```

### Models + Selection Criteria
Used SurrogateModels are defined by the array ```available_surrogate_models```.\
For each SurrogateModel, the dictionary ```available_selection_cirteria``` defines, which SelectionCriteria to use.

### Parallelization
End of file "_experiment_pipeline/experiment_pipeline_runner.py" distinguishes between running in parallel vs. executing in a loop. The used decision strategy looks at the ```mp.cpu_count()```, and runs in parallel if more than 32 cores are available.


## Single Experiment Run
Each experiment is executed in file "_experiment_pipeline/single_experiment_run.py". \
It first runs a grid search to find the best hyperparameters and afterwards runs for best_k hyperparameter combination the corresponding experiments. 
In function "get_evaluation_metrics(..)" for each used SurrogateModel, the set of Evaluation metrics to use is defined. 

## Grid Search
File "gridsearch/paramaeter_gridsearch.py" with function "get_best_parameter(..)" defines the grid search to execute. The file provides variables at the beginning, that defines the number of parameters to test, as well as the learning steps. For Grid Search, the csv-files folder "grid" are used. Function "get_parameter_grid(..)" returns for each Surrogate model the dictionary with a set of hyperparameters. 

NOTE: I did not used grid search for continuous experiments, but reused results from discrete results. However, it "should" work. 

## Evaluation Criteria
Several EvaluationCriteria are available, listed in folder "evaluation". The metric assigning function provides for each data SurrogateModel a set of Metrics. The best ones to use are "gp_uncertainty.py" for Gaussian Processes and "svdd_uncertainty" for SVDD. They log for each data point after each Iteration the uncertainty scoring. Thus, evaluation can be executed afterwards. However, the file tends to be very large!

Outputs are stored in ".\output", sorted by "mode\data_set\surrogate_model\selection_criteria\best-k_sample_log"
