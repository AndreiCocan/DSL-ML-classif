import { Model, Data, Algo, Trainer,SVM, isSVM } from '../language/generated/ast.js';
import * as fs from 'node:fs';
import { CompositeGeneratorNode, NL, toString } from 'langium';
import { extractDestinationAndName } from './cli-util.js';
import path from 'node:path';


export function generateClassifierPython(model: Model, filePath: string, destination: string | undefined, fileNode: CompositeGeneratorNode) {

    
    const data = extractDestinationAndName(filePath, destination);
    const generatedFilePath = `${path.join(data.destination, data.name)}.py`;
    
    fileNode.append('from sklearn import *', NL);
    fileNode.append('import pandas as pd', NL);
    fileNode.append(NL);

    generateData(model.all_data,fileNode);
    generateAlgos(model.all_algos,fileNode);
    generateTrainers(model.all_trainers,fileNode);

    if (!fs.existsSync(data.destination)) {
        fs.mkdirSync(data.destination, { recursive: true });
    }
    fs.writeFileSync(generatedFilePath, toString(fileNode));
    
    return generatedFilePath;
}

function generateData(data: Data[],fileNode: CompositeGeneratorNode) { 
    data.forEach((d,index) =>{

        //data.source: string
        fileNode.append(d.name,' = pd.read_csv("',d.source,'")', NL, NL);

        //data.label: string
        if (d.label != null){
            fileNode.append(d.name,'_Y',' = ',d.name,'["',d.label!,'"]',NL);
            d.drop.push(d.label!)

        }else{
            fileNode.append(d.name,'_Y',' = ',d.name,'.iloc[:,-1]',NL);
            fileNode.append(d.name,' = ',d.name,'.iloc[:, :-1]',NL, NL);
        }

        //data.drop: Array<string>
        if (d.drop.length>0){
            fileNode.append(d.name,' = ',d.name,'.drop(columns=["',d.drop.join('", "'),'"])',NL, NL);
        }

        //data.scaler: string
        if (d.scaler != null){
            fileNode.append(d.name,'_scaler = preprocessing.',d.scaler!,'Scaler()',NL);
            fileNode.append(d.name,' = ',d.name,'_scaler.fit_transform(',d.name,')',NL, NL);
        }
    })

}


function generateAlgos(algos: Algo[], fileNode: CompositeGeneratorNode){
    algos.forEach((algo,index) => {
        if(isSVM(algo)) generateSVM(algo,fileNode);
    })
}

function generateSVM(svm: SVM, fileNode: CompositeGeneratorNode){
    fileNode.append(svm.name, ' = svm.SVC(');
    var args_number = 0;
    
    //smv.kernel: string
    if(svm.kernel != null){
        fileNode.append('kernel = "',svm.kernel!,'"');
        args_number ++;
    }

    //svm.C: float
    if(svm.C != null){
        if(args_number>0) fileNode.append(', ');
        fileNode.append('C = ',svm.C!);
        args_number ++;
    }

    fileNode.append(')',NL, NL);
}

function generateTrainers(trainers: Trainer[],fileNode: CompositeGeneratorNode) { 
    trainers.forEach(trainer => {
        
        fileNode.append(trainer.data_ref.name,'_X_train, ', trainer.data_ref.name,'_X_test, ', trainer.data_ref.name,'_Y_train, ', trainer.data_ref.name,'_Y_test = model_selection.train_test_split(', trainer.data_ref.name,', ', trainer.data_ref.name, '_Y, ', 'test_size = ',trainer.train_test_split,')',NL);
        fileNode.append(trainer.algo_ref.name, '.fit(',trainer.data_ref.name,'_X_train',', ',trainer.data_ref.name,'_Y_train)', NL, NL);

        if(trainer.show_metrics){
            fileNode.append('print("Accuracy score : " + str(metrics.accuracy_score(',trainer.data_ref.name,'_Y_test, ',trainer.algo_ref.name,'.predict(',trainer.data_ref.name,'_X_test))))')
        }
    })  
}

