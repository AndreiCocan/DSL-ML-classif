# DSL-ML-classif

## Introduction 

The purpose of this report is to provide an overview of the development process and key features of our Domain-Specific Language (DSL) designed for machine learning classification. 
Our DSL is equipped with two compilers, one in Python and one in R.
This report explores the following key aspects of the project :
- The supported libraries and languages
- Abstract syntax of our DSL
- Examples of programs in action 
- The instructions to run our project
- The assessment of our different compilers 
- Our user feedback on Langium
- Our usage of LLM in the project


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
   source = "C:/..."
   label = "myClassToPredict" // if not specified, the last column of data will be taken as label
}
```
This program is valid but won't do anything as we don't ask to train any model.

A complete program
```
data myData {
	source = "C:/helloData"
        label = "myClassToPredict"
	drop = "unusedFeature1" "unusedFeature2"
	scaler = minMax
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
	drop = "unusedFeature1" "unusedFeature2"
	scaler = MinMax
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
**Prerequisites:** 
- docker


To run the project, the easiest way is to build and run NeoML in docker containers. 
To do so, go at the root of the repository and execute the following commands:

```bash
docker build -t NeoMLgenerator .
docker run --rm -it --entrypoint "/bin/bash" NeoMLgenerator
```
To run the NeoML generator in the docker container, use the following command:
```bash
./NeoML/bin/cli.js generate -d <Output Dir> -l <Python or R> <.neoml file> 
```

To run the tests for the generator in the docker container, use the following commands:
```bash
cd NeoML
npm test
```

To run the fuzzer to generate NeoML files and then execute the translated R and python scripts in the docker container, use the following commands:

```bash
cd fuzzer
./fuzzer_generate-and-compile.sh <Number of .neoml files generated> clean compile run
```

If you only want to generate NeoML files in the docker container, use the following commands:

```bash
cd fuzzer
python3 fuzzer.py 
```

To generate or update the benchmark, use the following command:

```bash
cd benchmark
python benchmark.py
```
This will compile and execute all the neoml files in ./Program_examples starting with "complete_"  and save the execution times and results in output.csv 



## Assessment of different compilers 




## Langium feedback

**1. Time Efficiency:**
Langium significantly expedited DSL development, allowing us to focus on language features instead of infrastructure, resulting in substantial time savings compared to starting from scratch.

**2. Playground Challenges:**
While the Langium playground aids grammar definition, frequent bugs required regular page refreshes, causing notable disruptions.

**3. Validation Testing Documentation:**
Improved documentation, particularly for validation testing, would enhance the learning curve. The tutorial provided by the teacher was necessary to streamline the development of these tests. 

**4. Synergy with AI:**
Despite the AI surge, DSLs remain powerful for expressing domain-specific concepts. Combining Langium with AI technologies offers a versatile toolkit, addressing different challenges in software development effectively.

In conclusion, Langium's time-saving advantages make it a valuable asset in DSL development, though addressing playground issues and enhancing specific documentation areas could further improve its usability. The coexistence of Langium and AI technologies provides developers with a holistic approach to diverse software development challenges.


## Usage of LLM in the project 

We leveraged ChatGPT for various aspects of DSL development:
- In the domain analysis phase, ChatGPT assisted us in structuring our analysis and refining our understanding.
- During the grammar-writing process, ChatGPT proved valuable in spotting and correcting errors (especially when we were facing playground bugs).
- In the development of our Fuzzer, ChatGPT was initially utilized to generate a foundational codebase by providing it with our grammar. Subsequently, we dedicated effort to fine-tune and optimize the generated code.
- However, for testing purposes, we did not utilize ChatGPT.
