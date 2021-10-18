# Customer churn prediction

### Introduction
For any business, customers are the basis for its success and that is why companies become more aware of the importance of gaining customer satisfaction. Customer retention includes all actions taken by an organization to guarantee customer loyalty and reduce customer churn. Customer churn refers to customers moving to a competitive organization or service provider.

### Task
The task for this 48h challenge is to analyze data and get to some conclusions regarding how the customer retention rate can be improved.

### Solution
There are various approaches to this problem that may depend more or less on the company's business plan. The two most popular approaches are customer segmentation and customer churn rate prediction. In this paper, the approach by which the mentioned problem was attacked is the prediction of customer churn. With a successful prediction of customer churn, it is possible to register customers before canceling their reservation and thus take appropriate actions that will keep the customers. Specifically, this sub-problem is known in the literature as Voluntary Churn (canceled) and/or Involuntary Churn (rejected).
The solution is based on creating a machine learning model for customer churn prediction. A successful machine learning model should predict with as much accuracy as possible whether a reservation will be canceled or rejected based on customer and product information.

### Process
The dataset for machine learning model training and testing was created by carefully selecting reservations from a database that contains complete information. Then, the dataset was cleaned of the missing values and the appropriate attributes were selected. After that, the basic descriptive analysis was applied to the selected attributes. The chosen machine learning model is Random Forest, which is trained in the manner of multiclass classification. Based on the complete reservation information, the model predicts whether the reservation is most likely to be successful, canceled, or rejected. The model is trained on 9750 instances of reservations, while the test data consists of 3250 instances of reservations. 

### Results
The model achieved results of 80% accuracy (F1 score: 76.86%) on the test set, which can be interpreted as the probability for predicting the outcome of the reservation. The attribute that has the greatest impact on the decision-making model is “instantly booked”, which is by far the most dominant with over 25% of the contribution. It can be seen that the feature “price” is not important for the model in making decisions. While the choice of package is a about 15% criterion whether the reservation will be successful, canceled, or rejected.

### Improvements
The model can be improved by using a larger sample of data, cleaning noises in the data, as well as detailed error analysis, etc. Interesting descriptive analyzes can also be done, such as: selection of users who contributed the most, Pearson's correlation coefficient between the class's attribute and other attributes etc.

### Experimental details

Dataset is split into a training set and test set. The training set is used in a 5-fold cross-validation technique for model optimization. On the other side, the testing set is used in the holdout procedure. Some of the „done“ reservations are dropped, because of the dataset imbalance state. The distribution of features was examined visually using histograms, KDE methods, as well as box-plot that indicate data inconsistencies.

Dataset split:

    Training set: 75.00% (9750)
    Distribution:
        * canceled 4226
        * declined 789
        * done 4735

    Testing set: 25.00% (3250)
    Distribution:
        * canceled 1408
        * declined 263
        * done 1579

Random forest model:

                precision recall f1-score support
    canceled       0.89    0.69    0.78    1408
    rejected       0.67    0.74    0.71    263
    done           0.76    0.90    0.82    1579

    accuracy        0.80  with support 3250          
    macro avg       0.77  with support 3250  
    weighted avg    0.79  with support 3250

    Total prediction time:  0.0488s
    Single instance:        0.00001502s
    Instances per second:   66557.73
