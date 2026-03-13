# Clinical-Stroke-Prediction
This project uses a **Decision Tree Classifier** to predict the likelihood of a stroke based on clinical features like age, glucose levels, and BMI. The focus is on **interpretability** and handling **imbalanced datasets**
##  Balanced Learning
Stroke datasets are often imbalanced (fewer stroke cases than healthy ones). This project utilizes the `class_weight='balanced'` parameter and the `entropy` criterion to ensure the model identifies minority class cases effectively.

##  Features
* **Interpretable AI:** The model is restricted to a `max_depth` of 3 to prevent overfitting and allow for a visualizable tree.
* **Visual Output:** Generates a detailed clinical flow chart (Decision Tree) showing the logic behind each prediction.
* **Data Imputation:** Handles missing BMI values using median imputation to maintain data integrity.

##  Tech Stack
* Python, Scikit-Learn, Pandas, Matplotlib,Decision Trees
