# DSL-ML-classif

## Introduction 

The purpose of this report is to provide an overview of the development process and key features of our Domain-Specific Language (DSL) designed for machine learning classification. 
Our DSL is equipped with two compilers, one in Python and the other in R.
This report explores the following key aspects of the project :
- The supported libraries and languages
- Abstract syntax of our DSL
- Examples of programs in action 
- The instructions to run our project
- The assessment of our different compilers 
- Our user feedback on Langium 


## Supported libraries and languages

The languages supported by our DSL are Python, utilizing the Scikit-learn library, and R, using several libraries like e1071, rpart, class, caret. 
These choices were made to leverage the extensive capabilities and popularity of Scikit-learn in Python for machine learning tasks and to harness the robustness and versatility of R's libraries.

In addition to Python and R, we also considered several other languages to be supported by our DSL. Our evaluation encompassed languages such as Java, C++, Go, and Scala. Each of these languages was thoroughly examined for their suitability in the context of our DSL, taking into account factors like performance and existing libraries (TensorFlow, DLIB, Shark, MLPack, GoLearn, Apache Spark MLlib).

(cf. Domain-Analysis.md)

## Abstract syntax

### Selected features for our DSL

**Tasks**
- Load data
- Scale data
- Define parameters for an algo
- Train a model (specifying algo and data to use)
- Visualize the results

**Entry data**
- Tabular

**Use case scenarios**
- Train a model and visualize the results
- Train several models that may reference different or same data and algos (that are previously defined)

### Metamodel

![](metamodel.png)

A trainer has a Data block and an Algo block. Every algorithm available for model training inherits from the Algo class. Our emphasis lies on SVM, KNN, Decision Tree, and neural networks, offering substantial classification possibilities, considering their adjustable parameters.

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

### A Critical Examination of Langium: Challenges in DSL Development

#### Introduction

Embarking on the journey of crafting a Domain-Specific Language (DSL) for machine learning classification using Langium, an Eclipse-based framework, proved to be a mixed experience. While the tool demonstrated commendable attributes such as speed and user-friendly features, several notable shortcomings surfaced during our DSL development.

#### Long-Term Viability of DSLs

Expanding on the concern about the future relevance of DSLs, it's crucial to consider the evolving nature of software development practices. The advent of low-code and no-code platforms, coupled with the growing sophistication of AI, raises questions about the enduring importance of DSLs. Teams investing in DSL development tools like Langium should carefully evaluate whether the benefits they provide justify the effort, especially in comparison to emerging alternatives.

Additionally, exploring ways in which AI can generate, complement and enhance DSLs could be a promising avenue for future development. Integrating DSLs with AI-driven solutions may present new opportunities for combining the strengths of both approaches.
