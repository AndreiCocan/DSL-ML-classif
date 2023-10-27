import type { Model, Data, Algo, Trainer } from '../language/generated/ast.js';
import * as fs from 'node:fs';
import { CompositeGeneratorNode, NL, toString } from 'langium';
import * as path from 'node:path';
import { extractDestinationAndName } from './cli-util.js';

export function generateClassifier(model: Model, filePath: string, destination: string | undefined): string {

    
    const data = extractDestinationAndName(filePath, destination);
    const generatedFilePath = `${path.join(data.destination, data.name)}.py`;
    
    const fileNode = new CompositeGeneratorNode();

    
    
    fileNode.append('import sklearn import *', NL);
    fileNode.append('import panda import *', NL);

    const all_data: Object[] = generateData(model.all_data,);
    const all_trainers: Object[] = generateTrainers(model.all_trainers,model.all_algos,fileNode);

    all_data;
    all_trainers;

    if (!fs.existsSync(data.destination)) {
        fs.mkdirSync(data.destination, { recursive: true });
    }
    fs.writeFileSync(generatedFilePath, toString(fileNode));
    return generatedFilePath;
}


function generateData(data: Data[]): Object[] { 
    return data

}


function generateTrainers(trainers: Trainer[],algos: Algo[],fileNode: CompositeGeneratorNode): Object[] { 
    trainers.forEach(trainer => generateTrainerBlock(trainer,fileNode))
    
    return trainers

}

function generateTrainerBlock(trainer: Trainer, fileNode: CompositeGeneratorNode) {

    fileNode.append('X_train, X_test, y_train, y_test=  train_test_split(X, Ynpm run build , test_size=',trainer.train_test_split,')',NL)
}