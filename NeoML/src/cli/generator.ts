import type { Model, Data, Algo, Trainer } from '../language/generated/ast.js';
import * as fs from 'node:fs';
import { CompositeGeneratorNode, NL, toString } from 'langium';
import * as path from 'node:path';
import { extractDestinationAndName } from './cli-util.js';

export function generateClassifier(model: Model, filePath: string, destination: string | undefined): string {

    
    const data = extractDestinationAndName(filePath, destination);
    const generatedFilePath = `${path.join(data.destination, data.name)}.py`;
    
    const fileNode = new CompositeGeneratorNode();

    
    
    fileNode.append('import sklearn', NL);
    fileNode.append('import pandas as pd', NL);
    fileNode.append(NL);

    generateData(model.all_data,fileNode);
    const all_trainers: Object[] = generateTrainers(model.all_trainers,model.all_algos,fileNode);


    all_trainers;

    if (!fs.existsSync(data.destination)) {
        fs.mkdirSync(data.destination, { recursive: true });
    }
    fs.writeFileSync(generatedFilePath, toString(fileNode));
    return generatedFilePath;
}


function generateData(data: Data[],fileNode: CompositeGeneratorNode) { 
    data.forEach((d,index) =>{

        //data.source: string
        fileNode.append(d.name,' = pd.readcsv("',d.source,'")', NL, NL);

        //data.label: string
        if (d.label != null){
            fileNode.append(d.name,'_Y',' = ',d.name,'[',d.label!,']',NL);
            d.drop.push(d.label!)

        }else{
            fileNode.append(d.name,'_Y',' = ',d.name,'.iloc[:,-1:]',NL);
            fileNode.append(d.name,' = ',d.name,'.iloc[:, :-1]',NL, NL);
        }

        //data.drop: Array<string>
        if (d.drop.length>0){
            fileNode.append(d.name,' = ',d.name,'.drop(columns=["',d.drop.join('", "'),'"])',NL, NL);
        }

        //data.scaler: string
        if (d.scaler != null){
            fileNode.append(d.name,'_scaler = sklearn.',d.scaler!,'Scaler()',NL);
            fileNode.append(d.name,' = ',d.name,'_scaler.fit_transform(',d.name,')',NL, NL);
        }
    })

}


function generateTrainers(trainers: Trainer[],algos: Algo[],fileNode: CompositeGeneratorNode): Object[] { 
    trainers.forEach(trainer => generateTrainerBlock(trainer,algos,fileNode))
    
    return trainers

}


function generateTrainerBlock(trainer: Trainer,algos: Algo[], fileNode: CompositeGeneratorNode) {
    
    //data.train_test_split: string
    var split_percent = trainer.train_test_split != null ? trainer.train_test_split : 'None';
    fileNode.append('X_train, X_test, y_train, y_test =  sklearn.train_test_split(X, Y , test_size = ',split_percent,')',NL);



}