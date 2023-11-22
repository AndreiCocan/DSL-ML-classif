# DSL-ML-classif

## Introduction 

The purpose of this report is to provide an overview of the development process and key features of our Domain-Specific Language (DSL) designed for machine learning classification. 
Our DSL is equipped with two compilers, one in Python and the other in R.
This report explores the following key aspects of the project :
- The supported libraries and languages
- Abstract syntax of our DSL
- Examples of programs in action 
- The instructions to run our language project
- The assessment of our different compilers 
- Our user feedback on Langium 


## Supported libraries and languages

The languages supported by our DSL are Python, utilizing the Scikit-learn library, and ???. 
These choices were made to leverage the extensive capabilities and popularity of Scikit-learn in Python for machine learning tasks and ???.


## Abstract syntax

### Expected features for our DSL

**Tasks**
- Load data
- Scale data
- Load a trained model
- Train a model (with or without cross-validation)
- Visualize the results

**Entry data**
- Tabular

**Use case scenarios**
- Train a model and visualize the results
- Use a trained model (to make predictions) and visualize the results

### Metamodel

![](metamodel.png)


## Programs in action 

Very simple program
```
data myData {
   source = "C:/...";
   label = "myClassToPredict"; // if not specified, the last column of data will be taken as label
}
```
This program is valid but won't do anything as we don't ask to train any model.

A complete program
```
data myData {
	source = "C:/helloData"
        label = "myClassToPredict"
	drop = ["unusedFeature1", "unusedFeature2"];
	scaler = minMax;
}

data myData2 {
	source = "C:/holaData"
}
         
algo mySvmModel svm {
	C = 0.0
	kernel = sigmoid
}

algo myKnnModel knn {
	n_neighbors = 8
	weights = distance
}
         
trainer {
	data = data.myData
        model = algo.mySvmModel
        train_test_split = 0.7
}
```
Only the referenced data and algo blocks in a trainer will be used: in this case the myData and mySvmModel blocks. Blocks myData2 and myKnnModel aren't referenced in any trainer so they won't be taken into account.

A more complex program that train several models
```
data myData {
	source = "C:/helloData"
        label = "myClassToPredict"
	drop = ["unusedFeature1", "unusedFeature2"];
	scaler = minMax;
}

data myData2 {
	source = "C:/holaData"
}
         
algo mySvmModel svm {
	C = 0.0
	kernel = sigmoid
}

algo myKnnModel knn {
	n_neighbors = 8
	weights = distance
}
         
trainer {
	data = data.myData
        model = algo.mySvmModel
        train_test_split = 0.7
        show_metrics = true
}

trainer {
	data = data.myData2
	model = algo.myKnnModel
}
```
This time, myData2 and myKnnModel are referenced in a second trainer, so they will be used. Here, only the metrics about the SVM model will be printed.


## How to run the project



## Assessment of different compilers 



## Langium 
