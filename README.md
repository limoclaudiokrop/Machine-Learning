# Machine Learning
The project looks to develop suitable predictive model for house prices in Melbourne. The project investigates performance of 
three machine learning models i.e Decision Trees, Random forest and XGBoost. For more specific details about code implementation
and performance, please refer to the jupyter notebook file (ipynb). Below is more information about the individual models.

## Decision Tree:
Decision Trees are versatile and intuitive machine learning models used for both classification and regression tasks. They make predictions by recursively partitioning the feature space into segments that homogenize the target variable. Each partition is represented by a node, and the tree branches out based on the values of different features. These models are easy to interpret, visually understandable, and can handle both numerical and categorical data without much preprocessing. However, they are prone to overfitting, especially when they grow to be very deep and complex. Techniques like pruning, limiting tree depth, or using ensemble methods help mitigate this issue.

## Random Forest:

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training. Each tree in the forest is built using a random subset of the training data and a random subset of features. When making predictions, the random forest aggregates the predictions from each individual tree and outputs the average (for regression) or the majority vote (for classification). Random Forests are less prone to overfitting compared to individual decision trees due to their nature of combining multiple trees. They excel in handling high-dimensional datasets, are less sensitive to outliers, and provide feature importance measures. They are widely used in various applications due to their robustness and effectiveness.

## XGBoost:

XGBoost, short for eXtreme Gradient Boosting, is an advanced implementation of gradient boosting machines. It is known for its high performance and efficiency in predictive modeling. XGBoost works by building an ensemble of weak learners (usually decision trees), where each new tree corrects the errors made by the previously trained models. It optimizes a specific objective function, using gradient descent algorithms to minimize the loss. XGBoost introduces regularization terms to control overfitting, supports parallel computing to improve speed, and handles missing values internally. It also provides flexibility in defining custom optimization objectives and evaluation criteria. XGBoost has been a winning algorithm in many machine learning competitions and is widely used in various domains due to its exceptional predictive power.

## Comparison:

Decision trees are the building blocks for both Random Forest and XGBoost. While decision trees are simple and interpretable, they tend to overfit. Random Forests address this by creating an ensemble of trees, reducing variance and improving accuracy. On the other hand, XGBoost further enhances model performance by sequentially building trees and optimizing a specific objective, leading to better predictive accuracy and speed compared to traditional gradient boosting. XGBoost's regularization techniques help control overfitting and make it less susceptible to variance.

In essence, Decision Trees provide an intuitive understanding of data, Random Forests improve prediction accuracy by leveraging multiple trees, and XGBoost pushes the boundaries of performance optimization in gradient boosting, making it a powerful choice for many machine learning tasks. Each method has its strengths and is suitable depending on the problem at hand, the dataset characteristics, and the trade-offs between interpretability and predictive performance.

## Conclusion
In conclusion the results from the melbourn housing data set reflected the expected perfromance of these models. Decision Trees performed the worst
 while Random Forest and XGBoost performed better. 
