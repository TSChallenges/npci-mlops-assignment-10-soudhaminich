# NPCI MLOps Assignment-10
## Kubeflow Pipeline for Customer Churn Prediction Model Training [10 Marks]

###  Overview
This assignment is to implement  **Kubeflow pipeline** to process **Customer Churn data** and train a machine learning model. 

The pipeline includes the flowing components:  
- **Data Loading**: Loading raw data.  
- **Train Test split**: Splitting data into train and test sets.
- **Model Training**: Training a classification model.
- **Model Evaluation**: Getting the model's performance metrics.

### Dataset Description:
The dataset you'll be working with is a customer dataset from a **Credit Card company**, which includes the following features:

- **RowNumber:** corresponds to the record (row) number and has no effect on the output.
- **CustomerId:** contains random values and has no effect on customer leaving the bank.
- **Surname:** the surname of a customer has no impact on their decision to leave the bank.
- **CreditScore:** can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
- **Geography:** A customer’s location can affect their decision to leave the bank.
- **Gender:** It’s interesting to explore whether gender plays a role in a customer leaving the bank.
- **Age:** This is certainly relevant since older customers are less likely to leave their bank than younger ones.
- **Tenure:** refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
- **Balance:** is also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
- **NumOfProducts:** refers to the number of products that a customer has purchased through the bank.
- **HasCrCard:** denotes whether or not a customer has a credit card. This column is also relevant since people with credit card are less likely to leave the bank.
- **IsActiveMember:** Active customers are less likely to leave the bank.
- **EstimatedSalary:** as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
- **Exited:** whether or not the customer left the bank. (0=No,1=Yes)

---
 
## Assignment Tasks:

**Note: Please refer to the lab guide material on Kubeflow Pipelines, for all the commands needed to complete the following tasks.**

**1. Understand the given files**

  You are provided with the following:
  * **Model training script (`train_model.py`)**
  * **Kubeflow Pipeline compilation script (`pipeline.py`)**


**2. Create a Codespace**
* Create a GitHub Codespace using the repository with default selection for Branch and Region, for Machine type select **4-core**. You can also change the Machine type to 4-core even after starting the Codespace.

**3. Setting up Kubernetes cluster** [2 Marks]
* Set up a single-node Minikube on your Codespace.
* Switch to root and start Minikube.
* Install kubectl to communicate with the Minikube cluster.

**4. Installing Kubeflow Pipelines** [2 Marks]
* Install Kubeflow pipelines using the manifest files from the Git repository.
* Verify pod creation on the cluster and accessibility to the Kubeflow Pipeline UI dashboard by port-forwarding.

**5. Creating a Kubeflow Pipeline and running it** [6 Marks]
* Complete the file stub `pipeline.py` script that contains components and pipeline to train the customer churn prediction model.
* Execute the Python file to compile pipeline to an intermediate representation YAML file.
* Create a new pipeline on Kubeflow UI by uploading the YAML file.
* Execute the new pipeline.


## Submission Guidelines
After completing the assignment by running the Kubeflow pipeline successfully, submit screenshots of your executions and commands in the folder `SubmissionImages` and then,

  - Stage your changes and commit the files:
    ```
    git add .
    git commit -m "assignment completed "
    ```
  - Push your changes to the GitHub repository:
    ```
    git push
    ```

Good luck, and happy coding!
