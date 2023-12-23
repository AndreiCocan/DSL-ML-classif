# Fuzzer for NeoML

## Introduction
The second assignment of this project consisted in a follow-up project for which we had to choose the subject among a defined list. We chose to develop a generator of random correct programs written in our DSL. We can see this generator as a fuzzer, as the goal is to assess our compilers over a great number of programs. The main challenge lied in generating programs syntactically and semantically correct.

## Project structure
The fuzzer consists of a Python script that is used in a Bash script to generate a given number of NeoML programs. This Bash script use our compilers to compile these programs in Python and R, and then run them.

To run this script in the docker context, do the following command in the `fuzzer` directory:
```sh
./fuzzer_generate-and-compile.sh 10 clean compile run
```
Replace '10' by any number of NeoML programs you want to generate.


Generated NeoML programs will be stored in `fuzzer/generated_programs`, Python and R associated programs will be stored in `fuzzer/compiled_programs`. By uncommenting the end of the two following lines in the Bash script `fuzzer_generate-and-compile.sh`:
```sh
python3 $file #&> ./run_results/$(basename "$file").txt
Rscript $file #&> ./run_results/$(basename "$file").txt
```
the outputs will be stored in text files in `fuzzer/run_results` directory, instead of being displayed in the terminal.


To be able to run the compiled programs, we provided a few real datasets in `datasets` directory that are also used for compilation tests.

## Outcomes

### Fix bugs in our DSL
The results of the fuzzer allowed us to fix a few bugs that we missed until this point through our manual tests, for example:
- handle the case where optional property 'train_test_split' of a trainer is null;
- add a new line after generating lines associated to a trainer in Python (the bug emerged with NeoML programs containing several trainers);
- fix the condition over the MLP's optional hidden_layer_sizes property;
- change incorrect scaler name 'AbsMax' in our grammar into 'MaxAbs'.

### Enhance features of our DSL
It also allowed us to enhance some features:
- drop missing values with both compilers as we encountered the case in the `titanic__Survived.csv` dataset we added for the fuzzer;
- suppress messages printed when R packages are loaded.

### Unresolved errors
From time to time, we also encountered 2 errors in R programs executions:
```
Error in terms.formula(formula, data = data) : 
  '.' dans la formule et pas d'argument 'data'
Calls: nnet ... <Anonymous> -> model.frame.default -> terms -> terms.formula
Exécution arrêtée
```
Which is very strange as the formula and arguments of the model function, in this case `nnet`, are correct (like in many other generated programs). After searching on the internet, it seems to be a bug, where sometimes R isn't able to interpret the point `.` as all the features. Giving explicitly all the features names may work but it would add several lines that would burden the code overall, only to handle some not very clear cases.
This issue is discussed here: https://github.com/dmcglinn/quant_methods/issues/22

```
Error in model.frame.default(formula = MbzodFMC_Y_train ~ ., data = MbzodFMC_X_train,  : 
  'data' must be a data.frame, not a matrix or an array
Calls: rpart ... eval.parent -> eval -> eval -> <Anonymous> -> model.frame.default
Exécution arrêtée
```
We tried to fix this error by explicitly use the function `data.frame` over the dataset before, but it didn't work. We thought for a while that it might occur because too much columns are dropped before, but decreasing the number of dropped columns didn't change anything either. After searching on the internet, no clear and evident solution emerged, as can be seen on this page: https://stackoverflow.com/questions/66361115/data-must-be-a-data-frame-not-a-matrix-or-an-array-error-in-r.


As these errors only appear sometimes and because of the lack of documented solutions, we haven't managed until this point to identify their possible causes. We decided to leave these two problems behind as there are quite marginal.

### Warnings

Some warnings may also be displayed from time to time, for example:
```
/home/username/.local/lib/python3.8/site-packages/sklearn/neural_network/_multilayer_perceptron.py:684: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
```
It can occur when the generated parameters for MLP are not relevant considering the dataset used, in our case, the hidden_layer_sizes property. It doesn't impede the execution though.

## Conclusion
The NeoML fuzzer project was a significant exploration into assessing our DSL compilers through random program generation. The setup involved a Python-based fuzzer script coordinated within a Bash script, generating NeoML programs for Python and R compilers. Using real datasets allowed us to run the obtained Python and R scripts, eventually uncovering bugs and enhancing some features. However, sporadic errors during R program execution persisted, evading clear resolution despite our investigations. Despite these challenges, the fuzzer project fulfilled its purpose as it substantially improved our DSL, and now allows us to affirm that the project is overall reliable.
