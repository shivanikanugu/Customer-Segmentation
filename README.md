 Customer Segmentation using Machine Learning 
===============================================


Run This App
------------
https://customer-segmentation-2zm698xavu8twwpj332pig.streamlit.app/

This repository contains the code and resources for a customer segmentation project using KMeans clustering algorithms. The project includes:

* Data preprocessing
* KMeans clustering
* Evaluation and visualization
* Deployment of a web application using Streamlit


Data Preprocessing
-----------------
  
![newplot (4)](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/newplot%20(4).png)
![newplot](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/newplot.png)
![newplot (1)](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/newplot%20(1).png)
![newplot (2)](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/newplot%20(2).png)
![newplot (3)](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/newplot%20(3).png)


Algorithms Analysis
-----------------

![algorithm](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/assets/295300023-61310abc-7a51-4041-ada2-f52e63d50c3b.png)



Deployment on Web
-----------------

![assets/Screenshot (171).png](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/assets/Screenshot%20(171).png)
![assets/Screenshot(172).png ](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/assets/Screenshot%20(172).png)
![assets/Screenshot(173).png ](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/assets/Screenshot%20(173).png)
![assets/Screenshot(174).png ](https://github.com/shivanikanugu/Customer-Segmentation/blob/main/assets/Screenshot%20(174).png)


Prerequisites
-------------

* Python 
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Streamlit

Getting Started
---------------

1. Clone this repository:

```bash
git clone https://github.com/shivanikanugu/Customer-Segmentation.git
```

2. Navigate to the project directory:

```bash
cd customer-segmentation
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Google Colab `Customer_Segmentation.ipynb` to perform data preprocessing, clustering, and evaluation.

5. To deploy the web application, run:

```bash
streamlit run _app.py
```

Data Preprocessing
------------------

The data preprocessing step involves:

* Loading the customer dataset
* Cleaning the data
* Feature engineering
* Scaling the data

KMeans Clustering
-----------------

KMeans clustering algorithm is used to segment the customers into different groups based on their features. The algorithm is run with different values of `k` to select the optimal number of clusters.

Evaluation and Visualization
----------------------------

The evaluation step involves:

* Visualizing the clusters in 2D and 3D
* Calculating the Within-Cluster Sum of Squares (WCSS) for each value of `k`
* Selecting the optimal number of clusters based on the WCSS plot

References
----------

* [KMeans Clustering in Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
* [Streamlit Documentation](https://docs.streamlit.io/)

License
-------

This project is licensed under the MIT License. See the `LICENSE` file for details.  
