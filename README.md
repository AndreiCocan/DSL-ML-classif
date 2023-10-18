# DSL-ML-classif

## Analyse du domaine

### Java

TensorFlow est une bibliothèque open source développée par Google qui est largement utilisée dans le domaine de l'apprentissage automatique (machine learning) et de l'apprentissage profond (deep learning). La version Java de TensorFlow permet aux développeurs Java de bénéficier de l'écosystème de TensorFlow tout en travaillant dans leur langage de prédilection.

Site officiel : https://www.tensorflow.org/jvm/install?hl=fr 

Exemple : https://github.com/tensorflow/java-models/blob/master/tensorflow-examples/src/main/java/org/tensorflow/model/examples/regression/linear/LinearRegressionExample.java
<details>
<summary>Open code example</summary>

```java
package org.tensorflow.model.examples.regression.linear;

import java.util.List;
import java.util.Random;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.framework.optimizers.GradientDescent;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Div;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.math.Pow;
import org.tensorflow.types.TFloat32;

/**
 * In this example TensorFlow finds the weight and bias of the linear regression during 1 epoch,
 * training on observations one by one.
 * <p>
 * Also, the weight and bias are extracted and printed.
 */
public class LinearRegressionExample {
    /**
     * Amount of data points.
     */
    private static final int N = 10;

    /**
     * This value is used to fill the Y placeholder in prediction.
     */
    public static final float LEARNING_RATE = 0.1f;
    public static final String WEIGHT_VARIABLE_NAME = "weight";
    public static final String BIAS_VARIABLE_NAME = "bias";

    public static void main(String[] args) {
        // Prepare the data
        float[] xValues = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f};
        float[] yValues = new float[N];

        Random rnd = new Random(42);

        for (int i = 0; i < yValues.length; i++) {
            yValues[i] = (float) (10 * xValues[i] + 2 + 0.1 * (rnd.nextDouble() - 0.5));
        }

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Define placeholders
            Placeholder<TFloat32> xData = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.scalar()));
            Placeholder<TFloat32> yData = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.scalar()));

            // Define variables
            Variable<TFloat32> weight = tf.withName(WEIGHT_VARIABLE_NAME).variable(tf.constant(1f));
            Variable<TFloat32> bias = tf.withName(BIAS_VARIABLE_NAME).variable(tf.constant(1f));

            // Define the model function weight*x + bias
            Mul<TFloat32> mul = tf.math.mul(xData, weight);
            Add<TFloat32> yPredicted = tf.math.add(mul, bias);

            // Define loss function MSE
            Pow<TFloat32> sum = tf.math.pow(tf.math.sub(yPredicted, yData), tf.constant(2f));
            Div<TFloat32> mse = tf.math.div(sum, tf.constant(2f * N));

            // Back-propagate gradients to variables for training
            Optimizer optimizer = new GradientDescent(graph, LEARNING_RATE);
            Op minimize = optimizer.minimize(mse);

            try (Session session = new Session(graph)) {

                // Train the model on data
                for (int i = 0; i < xValues.length; i++) {
                    float y = yValues[i];
                    float x = xValues[i];

                    try (TFloat32 xTensor = TFloat32.scalarOf(x);
                         TFloat32 yTensor = TFloat32.scalarOf(y)) {

                        session.runner()
                                .addTarget(minimize)
                                .feed(xData.asOutput(), xTensor)
                                .feed(yData.asOutput(), yTensor)
                                .run();

                        System.out.println("Training phase");
                        System.out.println("x is " + x + " y is " + y);
                    }
                }

                // Extract linear regression model weight and bias values
                List<?> tensorList = session.runner()
                        .fetch(WEIGHT_VARIABLE_NAME)
                        .fetch(BIAS_VARIABLE_NAME)
                        .run();

                try (TFloat32 weightValue = (TFloat32)tensorList.get(0);
                     TFloat32 biasValue = (TFloat32)tensorList.get(1)) {

                    System.out.println("Weight is " + weightValue.getFloat());
                    System.out.println("Bias is " + biasValue.getFloat());
                }

                // Let's predict y for x = 10f
                float x = 10f;
                float predictedY = 0f;

                try (TFloat32 xTensor = TFloat32.scalarOf(x);
                     TFloat32 yTensor = TFloat32.scalarOf(predictedY);
                     TFloat32 yPredictedTensor = (TFloat32)session.runner()
                             .feed(xData.asOutput(), xTensor)
                             .feed(yData.asOutput(), yTensor)
                             .fetch(yPredicted)
                             .run().get(0)) {

                    predictedY = yPredictedTensor.getFloat();

                    System.out.println("Predicted value: " + predictedY);
                }
            }
        }
    }
}
```
</details>

---

### C++

Pour les tâches de classification en apprentissage automatique, le C++ offre plusieurs avantages : Performance, Contrôle, Intégration. les bibliothèques les plus couramment utilisées étant : Tensorflow, Pytorch, DLIB, Shark, MLPack.
Shark est une bibliothèque open source de haute performance pour l'apprentissage automatique et l'optimisation, écrite en C++. Elle est conçue pour fournir une large gamme d'outils et d'algorithmes pour diverses tâches en apprentissage automatique, notamment la classification, la régression, le regroupement, la réduction de la dimensionnalité, et bien plus encore.

#### Exemple de la création et l'entraînement d’un réseau de neurones
Source : [exemple de la documentation shark-ml](http://image.diku.dk/shark/doxygen_pages/html/_f_f_n_n_basic_tutorial_8cpp_source.html ))
<details>
	<summary>Open code example for neural network</summary>

```cpp

#include <shark/Models/LinearModel.h>//single dense layer
#include <shark/Models/ConcatenatedModel.h>//for stacking layers, provides operator>>
//training the  model
#include <shark/ObjectiveFunctions/ErrorFunction.h>//error function, allows for minibatch training
#include <shark/ObjectiveFunctions/Loss/CrossEntropy.h> // loss used for supervised training
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h> // loss used for evaluation of performance
#include <shark/Algorithms/GradientDescent/Adam.h> //optimizer: simple gradient descent.
#include <shark/Data/SparseData.h> //loading the dataset
using namespace shark;

std::size_t batchSize = 256;
LabeledData<RealVector,unsigned int> data;
importSparseData( data, argv[1], 0, batchSize );
data.shuffle(); //shuffle data randomly
auto test = splitAtElement(data, 70 * data.numberOfElements() / 100);//split a test set
std::size_t numClasses = numberOfClasses(data);
std::size_t inputDim = inputDimension(data);

//We use a dense linear model with rectifier activations
typedef LinearModel<RealVector, RectifierNeuron> DenseLayer;

//build the network
DenseLayer layer1(inputDim,hidden1, true);
DenseLayer layer2(hidden1,hidden2, true);
LinearModel<RealVector> output(hidden2,numClasses, true);
auto network = layer1 >> layer2 >> output;

//create the supervised problem.
CrossEntropy<unsigned int, RealVector> loss;
ErrorFunction<> error(data, &network, &loss, true);//enable minibatch training

//optimize the model
std::cout<<"training network"<<std::endl;
initRandomNormal(network,0.001);
Adam<> optimizer;
error.init();
optimizer.init(error);
for(std::size_t i = 0; i != iterations; ++i){
        optimizer.step(error);
        std::cout<<i<<" "<<optimizer.solution().value<<std::endl;
}
network.setParameterVector(optimizer.solution().point);
```
</details>

#### Exemple d’un KNN
Source : [exemple de la documentation shark-ml](http://image.diku.dk/shark/doxygen_pages/html/_k_n_n_tutorial_8cpp_source.html ))

<details>
	<summary>Open code example for kNN</summary>

```cpp

 #include <shark/Data/Csv.h>
 #include <shark/Models/NearestNeighborModel.h>
 #include <shark/Algorithms/NearestNeighbors/TreeNearestNeighbors.h>
 #include <shark/Models/Trees/KDTree.h>
 #include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
 #include <shark/Data/DataView.h>
 #include <iostream>
 
 using namespace shark;
 using namespace std;
 
 int main(int argc, char **argv) {
  if(argc < 2) {
  cerr << "usage: " << argv[0] << " (filename)" << endl;
  exit(EXIT_FAILURE);
  }
  // read data
  ClassificationDataset data;
  try {
  importCSV(data, argv[1], LAST_COLUMN, ' ');
  }
  catch (...) {
  cerr << "unable to read data from file " << argv[1] << endl;
  exit(EXIT_FAILURE);
  }
 
  cout << "number of data points: " << data.numberOfElements()
  << " number of classes: " << numberOfClasses(data)
  << " input dimension: " << inputDimension(data) << endl;
 
  // split data into training and test set
  ClassificationDataset dataTest = splitAtElement(data, static_cast<std::size_t>(.5 * data.numberOfElements()));
  cout << "training data points: " << data.numberOfElements() << endl;
  cout << "test data points: " << dataTest.numberOfElements() << endl;
 
  //create a binary search tree and initialize the search algorithm - a fast tree search
  KDTree<RealVector> tree(data.inputs());
  TreeNearestNeighbors<RealVector,unsigned int> algorithm(data,&tree);
  //instantiate the classifier
  const unsigned int K = 1; // number of neighbors for kNN
  NearestNeighborModel<RealVector, unsigned int> KNN(&algorithm,K);
 
  // evaluate classifier
  ZeroOneLoss<unsigned int> loss;
  Data<unsigned int> prediction = KNN(data.inputs());
  cout << K << "-KNN on training set accuracy: " << 1. - loss.eval(data.labels(), prediction) << endl;
  prediction = KNN(dataTest.inputs());
  cout << K << "-KNN on test set accuracy: " << 1. - loss.eval(dataTest.labels(), prediction) << endl;
 }

```
</details>

#### Exemple de code l'entraînement d’un SVM avec cross validation
Source : [exemple de la documentation shark-ml](http://image.diku.dk/shark/doxygen_pages/html/_c_svm_grid_search_tutorial_8cpp_source.html ))

<details>
	<summary>Open code example for SVM with cross validation</summary>

```cpp

#include <shark/Models/Kernels/GaussianRbfKernel.h>
#include <shark/ObjectiveFunctions/Loss/ZeroOneLoss.h>
#include <shark/Algorithms/Trainers/CSvmTrainer.h>
#include <shark/Data/DataDistribution.h>

#include <shark/ObjectiveFunctions/CrossValidationError.h>
#include <shark/Algorithms/DirectSearch/GridSearch.h>
#include <shark/Algorithms/JaakkolaHeuristic.h>

using namespace shark;
using namespace std;

int main() {
    // problem definition
    Chessboard prob;
    ClassificationDataset dataTrain = prob.generateDataset(200);
    ClassificationDataset dataTest = prob.generateDataset(10000);

    // SVM setup
    GaussianRbfKernel<> kernel(0.5, true); // unconstrained?
    KernelClassifier<RealVector> svm;
    bool offset = true;
    bool unconstrained = true;
    CSvmTrainer<RealVector> trainer(&kernel, 1.0, offset, unconstrained);

    // cross-validation error
    const unsigned int K = 5; // number of folds
    ZeroOneLoss<unsigned int> loss;
    CVFolds<ClassificationDataset> folds = createCVSameSizeBalanced(dataTrain, K);
    CrossValidationError<KernelClassifier<RealVector>, unsigned int> cvError(
        folds, &trainer, &svm, &trainer, &loss
    );

    // find best parameters

    // use Jaakkola's heuristic as a starting point for the grid-search
    JaakkolaHeuristic ja(dataTrain);
    double ljg = log(ja.gamma());
    cout << "Tommi Jaakkola says gamma = " << ja.gamma() << " and ln(gamma) = " << ljg << endl;

    GridSearch grid;
    vector<double> min(2);
    vector<double> max(2);
    vector<size_t> sections(2);
    // kernel parameter gamma
    min[0] = ljg - 4.; max[0] = ljg + 4; sections[0] = 9;
    // regularization parameter C
    min[1] = 0.0; max[1] = 10.0; sections[1] = 11;
    grid.configure(min, max, sections);
    grid.step(cvError);

    // train model on the full dataset
    trainer.setParameterVector(grid.solution().point);
    trainer.train(svm, dataTrain);
    cout << "grid.solution().point " << grid.solution().point << endl;
    cout << "C =\t" << trainer.C() << endl;
    cout << "gamma =\t" << kernel.gamma() << endl;

    // evaluate
    Data<unsigned int> output = svm(dataTrain.inputs());
    double train_error = loss.eval(dataTrain.labels(), output);
    cout << "training error:\t" << train_error << endl;
    output = svm(dataTest.inputs());
    double test_error = loss.eval(dataTest.labels(), output);
    cout << "test error: \t" << test_error << endl;
}

```
</details>

---

### Python
Python est un langage de programmation interprété polyvalent, réputé pour sa simplicité syntaxique qui le rend facile à apprendre et à lire. Il est largement utilisé dans le développement web, la science des données, l'automatisation de tâches et bien d'autres domaines grâce à sa vaste bibliothèque standard et à sa communauté active de développeurs. Parmi les bibliothèques pour l’apprentissage automatique, on trouve Scikit-learn, Tensorflow et PyTorch.

#### Scikit-learn ([site web officiel](https://scikit-learn.org/stable/))
Scikit-learn est l'une des bibliothèques Python les plus populaires pour l'apprentissage automatique. Elle offre un large éventail d'outils et d'algorithmes pour le traitement des données, l'apprentissage supervisé et non supervisé, la réduction de la dimensionnalité, la sélection de modèles, l'évaluation des modèles, et bien plus encore. Scikit-learn est conçu pour être simple à utiliser, mais il reste puissant et flexible pour la résolution de tâches complexes en apprentissage automatique. Scikit-learn est utilisé dans divers domaines, tels que l'analyse de données, la classification d'images, la détection d'anomalies, la bioinformatique, la finance, et bien d'autres.

##### Exemple de code pour la reconnaissance de chiffres écrits à la main 
Source : [exemple de la documentation scikit-learn](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py))

<details>
	<summary>Open code example</summary>

```python
# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
```
##### Digits dataset
```python
digits = datasets.load_digits()

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
	ax.set_axis_off()
	ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
	ax.set_title("Training: %i" % label)
```
##### Classification
```python
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(
	data, digits.target, test_size=0.5, shuffle=False
)

# Learn the digits on the train subset
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
```
```python
# Visualization of the first 4 test samples and show their predicted digit value
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
	ax.set_axis_off()
	image = image.reshape(8, 8)
	ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
	ax.set_title(f"Prediction: {prediction}")
```
```python
print(
	f"Classification report for classifier {clf}:\n"
	f"{metrics.classification_report(y_test, predicted)}\n"
)
```
```python
# Confusion matrix
disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
```
</details>

---

### Go

Go, également connu sous le nom de Golang, est un langage de programmation open-source créé par Google. Il se distingue par sa simplicité syntaxique, son efficacité d'exécution et sa prise en charge native de la programmation concurrente avec les "goroutines". Go est largement utilisé dans le développement backend, les services web, les applications cloud et les outils en ligne de commande en raison de sa simplicité, de sa rapidité et de sa robustesse. Il offre également une gestion automatique de la mémoire, réduisant les erreurs liées à la gestion de la mémoire.

#### GoLearn

GoLearn ([repositoryl](https://github.com/sjwhitworth/golearn)) est une bibliothèque d'apprentissage automatique open source développée en langage de programmation Go (ou Golang). Elle offre un ensemble de fonctionnalités pour la création, l'entraînement et l'évaluation de modèles d'apprentissage automatique dans l’écosystème Go. 
#### Exemple de code
Source : [exemple de la documentation golang](https://golangdocs.com/golang-machine-learning-libraries))
<details>
	<summary>Open code example</summary>

```go
package main
 
import (
    "fmt"
 
    "github.com/sjwhitworth/golearn/base"
    "github.com/sjwhitworth/golearn/evaluation"
    "github.com/sjwhitworth/golearn/knn"
)
 
func main() {
    // Load in a dataset, with headers. Header attributes will be stored.
    // Think of instances as a Data Frame structure in R or Pandas.
    // You can also create instances from scratch.
    rawData, err := base.ParseCSVToInstances("datasets/iris.csv", false)
    if err != nil {
        panic(err)
    }
 
    // Print a pleasant summary of your data.
    fmt.Println(rawData)
 
    //Initialises a new KNN classifier
    cls := knn.NewKnnClassifier("euclidean", "linear", 2)
 
    //Do a training-test split
    trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)
    cls.Fit(trainData)
 
    //Calculates the Euclidean distance and returns the most popular label
    predictions, err := cls.Predict(testData)
    if err != nil {
        panic(err)
    }
 
    // Prints precision/recall metrics
    confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
    if err != nil {
        panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
    }
    fmt.Println(evaluation.GetSummary(confusionMat))
}

```
</details>

---

### Scala

Scala est un langage de programmation polyvalent conçu pour la concision et la sécurité. Il s'exécute sur la machine virtuelle Java (JVM) et offre un mélange de fonctionnalités orientées objet et fonctionnelles, ce qui en fait un choix populaire pour les applications de traitement des données et la création de systèmes évolutifs.

#### Apache Spark MLlib

Apache Spark MLlib est une bibliothèque d'apprentissage automatique open source conçue pour fonctionner avec le framework Apache Spark, en utilisant le langage de programmation Scala. L’intégration de la bibliothèque dans le framework Apache Spark permet de tirer parti de la mise en cluster et de la parallélisation pour traiter efficacement des volumes massifs de données.

<details>
	<summary>Open code example</summary>

```scala
import org.apache.spark.ml.classification.LinearSVC

// Load training data
val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

val lsvc = new LinearSVC()
  .setMaxIter(10)
  .setRegParam(0.1)

// Fit the model
val lsvcModel = lsvc.fit(training)

// Print the coefficients and intercept for linear svc
println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

```
</details>

---

### Tableau comparatif

<details>
	<summary>Open table</summary>

| Caractéristique                  | scikit-learn  (Python)     | GoLearn  (Go)          | TensorFlow (Java)         | Shark    (C++)           | MLlib     (Scala)          |
|----------------------------------|--------------------|--------------------|---------------------|---------------------|---------------------|
| Communauté et Support            | Grande communauté et support actif | Communauté en croissance | Communauté active   | Communauté en croissance | Communauté en croissance |
| Apprentissage automatique        | Oui, méthodes traditionnelles | Oui, axé sur les arbres de décision et les forêts aléatoires | Oui, large éventail de modèles | Oui, avec des bibliothèques pour l'optimisation numérique | Oui, diverses méthodes |
| Apprentissage en profondeur      | Non, sauf avec des extensions tierces (comme TensorFlow) | Non, principalement axé sur l'apprentissage automatique classique | Oui, avec des fonctionnalités complètes d'apprentissage en profondeur | Oui, avec des capacités d'apprentissage en profondeur | Non, principalement axé sur l'apprentissage automatique classique |
| Facilité d'utilisation            | Très convivial, idéal pour les débutants en ML | Convivial, mais moins de ressources disponibles pour les débutants | Un peu plus complexe, principalement destiné aux utilisateurs avancés | Convivial, mais peut nécessiter une expertise en C++ | Convivial, adapté aux utilisateurs de Scala |
| Flexibilité                      | Moins flexible en termes de personnalisation de modèles | Plus de flexibilité que scikit-learn, mais moins que TensorFlow | Très flexible avec la possibilité de personnaliser chaque aspect du modèle | Flexible avec une grande variété de paramètres personnalisables | Flexible avec des API haut niveau et bas niveau |
| Performances                     | Performances solides pour les tâches de base | Performances correctes, mais pas aussi performant que TensorFlow | Performances exceptionnelles, idéales pour l'apprentissage en profondeur | Performances solides pour l'apprentissage automatique classique | Performances solides pour diverses tâches |
| Déploiement                     | Facile à déployer en production grâce à sa simplicité | Peut être déployé, mais nécessite plus d'efforts que scikit-learn | Peut être déployé, mais nécessite une gestion plus complexe | Peut être déployé avec des efforts de déploiement C++ | Peut être déployé dans l'écosystème Spark |
| Cas d'utilisation typiques       | Exploration de données, classification, régression, clustering | Arbres de décision, forêts aléatoires, classification | Réseaux de neurones, traitement du langage naturel, vision par ordinateur | Apprentissage automatique classique, optimisation numérique | Diverses tâches d'apprentissage automatique |
</details>

## Expected features for our DSL

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

**Programs examples**

Very simple program
```
data "myData" {
   source = "C:/...";
   label = "myClassToPredict"; // if not specified, the last column of data will be taken as label
}

model "myFirstModel" svm {
   data_src = data.myData;
   // all svm parameters have default values
   // two parameters specific to our DSL are show_metrics (default value : true) and training_split (default value : 0.7)
}
```

More complex program
```
data "myData" {
   source = "C:/...";
   label = "myClassToPredict";
   drop = ["unusedFeature1", "unusedFeature2"];
   scaler = minMax;
}

model "myFirstModel" svm {
   data_src = data.myData;
}

model "mySecondModel" svm {
   data_src = data.myData;
   cross_validation = 5;
   // svm specific parameters
   C = 0.9;
   kernel = linear;
   gamma = auto;
}

model "myAlreadyTrainedModel" svm {
   data_src = data.myData;
   load = "C:/...";
}

// the metrics of all models will be printed
```


## Metamodel
