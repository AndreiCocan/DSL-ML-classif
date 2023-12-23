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

(cf. [Domain-Analysis.md](Domain-Analysis.md))


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
This will compile and execute all the neoml files in ./Program_examples starting with "complete_", plus 20 fuzzer-generated neoml programs, and save the execution times and results in output.csv 



## Assessment of different compilers 

In the benchmark folder, you'll discover the `benchmark.py` file, which operates on test NeoML programs and an additional 20 programs generated during execution using the fuzzer. The output and execution times of each script (Python and R for every NeoML program) are stored in a CSV file. Additionally, we compute the means and variances of execution times in both Python and R to facilitate a comparison between our two compilers. We've included sample output examples in the folder [](benchmark/examples_output_csv/). Our observations reveal consistent results across these executions.

In general, the compiled Python scripts consistently showcase faster execution times across various DSL programs, averaging around 0.76 seconds. It maintains a commendable accuracy, ranging from 0.32 to a perfect 1.0, demonstrating robust performance stability (considering we use sometimes a "random" dataset made to obtain bad accuracy).

Conversely, the R compiler consistently lags in speed, with execution times averaging about 1.70 seconds. Although its accuracy ranges between 0.49 to 1.0, it doesn't match the consistency observed in the Python compiler's performance.

Regardless of the dataset used, Python exhibits swifter execution, while the R compiler, albeit slower, showcases varied accuracy that occasionally matches or surpasses Python's accuracy. The choice between the two compilers depends on the priority: Python for faster execution or R for potential higher accuracy despite slower speeds.


## Fuzzer
Among all the proposed subjects, we chose to develop a fuzzer to generate valid random NeoML programs. This decision aimed to thoroughly test our DSL, identify bugs, and subsequently address them. The complementary part of the final assignment, detailing the obtained results for this follow-up project, is available in [Assignment2_fuzzer.md](Assignment2_fuzzer.md).


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
- In the development of our Fuzzer, ChatGPT was initially utilized to generate a foundational codebase by providing it with our grammar. Subsequently, we dedicated effort to adapt, fine-tune and optimize the generated code.
- However, for testing purposes, we did not utilize ChatGPT.


## Conclusion

Our language focuses specifically on machine learning classification tasks. We honed in on Python and R for their robust libraries in this domain. This language streamlines data handling, model training, and result visualization for classification purposes.

While Python consistently performed well, the development of our R compiler posed significant challenges due to unresolved bugs in the R libraries that remained obscure, as we detailed it in [Assignment2_fuzzer.md](Assignment2_fuzzer.md). With our benchmark, we compared the performance of our Python and R compilers—the obtained Python scripts are clearly and consistently quicker, but R scripts occasionally offer higher accuracy.

Langium played a key role in speeding up development, though it had some glitches along the way. ChatGPT (LLM) helped refine our approach during some phases of the development.

In essence, our language targets machine learning classification tasks, harnessing Python and R for their strengths in this area. Langium proved beneficial despite some hiccups, and we're optimistic about the potential collaboration with AI technologies.
