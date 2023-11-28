import { Model, Data, Algo, Trainer,SVM, KNN,MLP, isSVM, DecisionTree, isKNN, isMLP, isDecisionTree } from '../language/generated/ast.js';
import * as fs from 'node:fs';
import { CompositeGeneratorNode, NL, toString } from 'langium';
import { extractDestinationAndName } from './cli-util.js';
import path from 'node:path';
import chalk from 'chalk';


export function generateClassifierR(model: Model, filePath: string, destination: string | undefined, fileNode: CompositeGeneratorNode) {
    const data = extractDestinationAndName(filePath, destination);
    const generatedFilePath = `${path.join(data.destination, data.name)}.r`;

    fileNode.append('library(tidyr)',NL);
    fileNode.append('suppressMessages(library(dplyr))', NL);
    fileNode.append('library(e1071) # SVM implementation', NL);
    fileNode.append('library(class) # KNN implementation', NL);
    fileNode.append('library(rpart) # DecisionTree implementation', NL);
    fileNode.append('library(nnet) # MLP implementation', NL);
    fileNode.append('suppressMessages(library(caret))', NL,NL);

    generateData(model.all_data,fileNode);
    generateTrainers(model.all_trainers, model.all_algos, fileNode);

    if (!fs.existsSync(data.destination)) {
        fs.mkdirSync(data.destination, { recursive: true });
    }
    fs.writeFileSync(generatedFilePath, toString(fileNode));

    return generatedFilePath;
}


function generateData(data: Data[],fileNode: CompositeGeneratorNode) { 
    data.forEach((d,index) =>{

        //data.source: string
        fileNode.append(d.name,' <- read.csv("',d.source,'")', NL);
        fileNode.append(d.name,' <- ',d.name,' %>% drop_na()',NL,NL);

        //data.drop: Array<string>
        if (d.drop.length>0){
            fileNode.append('columns_to_drop <- c("',d.drop.join('", "'),'")', NL);
            fileNode.append(d.name,' <- ',d.name,' %>% select(-any_of(columns_to_drop))',NL, NL);
        }

        //data.label: string
        if (d.label != null){
            fileNode.append(d.name,'_X <- subset(',d.name,', select = -',d.label!,')', NL);
            fileNode.append(d.name,'_Y <- as.factor(',d.name,'$',d.label!,')',NL,NL);
        }else{
            fileNode.append(d.name,'_X <- subset(',d.name,', select = -ncol(',d.name,'))', NL);
            fileNode.append(d.name,'_Y',' <- as.factor(',d.name,'[,ncol(',d.name,')])',NL);
        }

        //data.scaler: string
        if (d.scaler != null){
            switch(d.scaler) {
                case "Standard":
                    fileNode.append(d.name,'_X <- scale(',d.name,'_X)',NL, NL);
                    break;
                case "MinMax":
                    fileNode.append('preprocess_range <- preProcess(',d.name,'_X, method = "range")', NL);
                    fileNode.append(d.name,'_X <- predict(preprocess_range, newdata = ', d.name,'_X)', NL, NL);
                    break;
                case "AbsMax":
                    fileNode.append(d.name, '_X <- apply(',d.name,'_X, 2, function(x) x / max(abs(x)))', NL, NL);
                    break;
            }
        }
    })

}


function generateTrainers(trainers: Trainer[], allAlgos: Algo[], fileNode: CompositeGeneratorNode) { 
    trainers.forEach(trainer => {
        const algo = allAlgos.find(a => (a.name == trainer.algo_ref.name));
        if(algo === undefined) {
            throw new TypeError('Algo not found');
        }

        if(trainer.train_test_split == null) {
            trainer.train_test_split = '0.7';
        }

        fileNode.append('train_index <- sample(1:nrow(',trainer.data_ref.name,'), (1-',trainer.train_test_split,') * nrow(', trainer.data_ref.name,'))', NL);
        // the partition is not like in scikit-learn
        fileNode.append(trainer.data_ref.name,'_X_train <- ', trainer.data_ref.name, '_X[train_index, ]', NL);
        fileNode.append(trainer.data_ref.name,'_X_test <- ', trainer.data_ref.name, '_X[-train_index, ]', NL);
        fileNode.append(trainer.data_ref.name,'_Y_train <- ', trainer.data_ref.name, '_Y[train_index]', NL);
        fileNode.append(trainer.data_ref.name,'_Y_test <- ', trainer.data_ref.name, '_Y[-train_index]', NL,NL);

        generateAlgo(algo, trainer.data_ref.name, (trainer.show_metrics === 'true'), fileNode);
    })  
}

function generateAlgo(algo: Algo, dataRefName: string, showMetrics: boolean, fileNode: CompositeGeneratorNode){
    if(isSVM(algo)) generateSVM(algo,dataRefName, showMetrics, fileNode);
    if(isKNN(algo)) generateKNN(algo,dataRefName, showMetrics, fileNode);
    if(isMLP(algo)) generateMLP(algo,dataRefName, showMetrics, fileNode);
    if(isDecisionTree(algo)) generateDT(algo,dataRefName, showMetrics, fileNode);
}

function generateSVM(svm: SVM, dataRefName: string, showMetrics: boolean, fileNode: CompositeGeneratorNode){
    fileNode.append(svm.name, ' <- svm(',dataRefName,'_X_train, ',dataRefName, '_Y_train');
    var args_number = 0;
    
    //svm.kernel: string
    if(svm.kernel != null){
        fileNode.append(', kernel = "',svm.kernel!,'"');
        args_number ++;
    }

    //svm.C: float
    if(svm.C != null){
        if(args_number>0) fileNode.append(', ');
        fileNode.append('cost = ',svm.C!);
        args_number ++;
    }

    fileNode.append(')',NL, NL);

    if(showMetrics) {
        generateShowMetrics(svm.name, dataRefName, fileNode);
    }
}

function generateKNN(knn: KNN, dataRefName: string, showMetrics: boolean, fileNode: CompositeGeneratorNode){
    fileNode.append(knn.name, ' <- knn(train = ', dataRefName,'_X_train, test = ', dataRefName, '_X_test, cl = ', dataRefName, '_Y_train');
    //var args_number = 0;
        
    //knn.n_neighbors: int
    if(knn.n_neighbors != null){
        fileNode.append(', k = ',String(knn.n_neighbors!));
        //args_number ++;
    }

    fileNode.append(')',NL);

    //knn.weights: string
    if(knn.weights != null && knn.weights == 'distance'){
        fileNode.append('# Not possible to use distance-based weights in R',NL);
    }
    fileNode.append(NL);

    if(showMetrics) {
        fileNode.append(knn.name, '_accuracy <- confusionMatrix(',knn.name,', ',dataRefName,'_Y_test)$overall["Accuracy"]',NL);
        fileNode.append('print(paste("',knn.name,' accuracy:", ',knn.name,'_accuracy))',NL,NL);
    }
}

function generateMLP(mlp: MLP, dataRefName: string, showMetrics: boolean, fileNode: CompositeGeneratorNode){
    fileNode.append(mlp.name, ' <- nnet(',dataRefName,'_Y_train ~ ., ', dataRefName, '_X_train');

    //var args_number = 0;
    
    //mlp.hidden_layer_sizes: int
    if(mlp.hidden_layer_sizes.length>0){
        fileNode.append(', size = ',String(mlp.hidden_layer_sizes![0]));
        //args_number ++;
    } else {
        fileNode.append(', size = ',String(8));
    }

    fileNode.append(')',NL, NL);

    if(showMetrics) {
        generateShowMetrics(mlp.name, dataRefName, fileNode);
    }
}

function generateDT(dt: DecisionTree, dataRefName: string, showMetrics: boolean, fileNode: CompositeGeneratorNode){
    fileNode.append(dt.name, ' <- rpart(',dataRefName,'_Y_train ~ ., ',dataRefName, '_X_train, method = "class"');
    
    var args_number = 0;

    //dt.criterion : string (function to measure the quality of a split)
    // In scikit-learn, supported criteria are “gini” for the Gini impurity and “log_loss” and “entropy” both for the Shannon information gain
    // Here in R, supported criteria are "gini" and "information" so we'll assume that log_loss and entropy will be translasted into information.
    if(dt.criterion != null){
        if(dt.criterion == 'gini') {
            fileNode.append(', parms = list(split = "',dt.criterion !,'")');
        } else {
            fileNode.append(', parms = list(split = "information")');
        }
        
        args_number ++;
    }

    //dt.max_depth : int
    if(dt.max_depth != null){
        if(args_number>0) fileNode.append(', ');
        fileNode.append(', control = rpart.control(maxdepth = ',String(dt.max_depth!),')');
        args_number ++;
    }

    //dt.splitter : string
    if(dt.splitter != null){
        console.log(chalk.yellowBright('No splitter parameter in R'));
    }

    fileNode.append(')',NL, NL);

    if(showMetrics) {
        generateShowMetrics(dt.name, dataRefName, fileNode);
    }
}

function generateShowMetrics(modelName: string, dataRefName: string, fileNode: CompositeGeneratorNode) {
    fileNode.append('pred <- predict(',modelName,', ',dataRefName,'_X_test, type = "class")',NL);
    fileNode.append('pred <- factor(pred, levels=levels(',dataRefName,'_Y_test))',NL);
    fileNode.append(modelName, '_accuracy <- confusionMatrix(pred, ',dataRefName,'_Y_test)$overall["Accuracy"]',NL);
    fileNode.append('print(paste("',modelName,' accuracy:", ',modelName,'_accuracy))',NL,NL);
}

