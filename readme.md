# pipeline-test
- `mlp.py`: MLP as regression
- `gpmodel.py`: gpmodel as regression
- `rf.py`: random forest as regression (nhid=5, also print emb corresponding to max train value, min train value, max eval value, min eval value)
- `classification.py` train a classification model. threshold: 0.75. print the confusion_matrix
- `classification_rf.py` train a classification model using rf. threshold: 0.75. print the confusion_matrix
- `multi-classification_rf.py` train a multi-classification model using rf. threshold: <0.69, <0.72, <0.75, >0.75 to make the training set class-balanced. print the confusion_matrix