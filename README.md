Machine Learning Project
========================

Run machine learning algorithms from the command line.
bin/run.py

```
mkvirtualenv machine_learning
python setup.py install
run.py --input-data-file "data/input/sample_data.csv" --number-features 2 --number-targets 1 --hypothesis-name "multiple_linear_regression" --cost-function-name "squared_error_loss" --algorithm-name "batch_gradient_descent" --learning-rate 0.0001 --tolerance .00000000000001
```
