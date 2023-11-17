import { describe, expect, test } from 'vitest';

import { SVM, type Model, KNN, isSVM, isKNN, DecisionTree, MLP, isDecisionTree, isMLP } from '../language/generated/ast.js';

import { AstNode, EmptyFileSystem, LangiumDocument } from 'langium';
import { parseDocument } from 'langium/test';
import { createNeoMlServices } from '../language/neo-ml-module.js';

const services = createNeoMlServices(EmptyFileSystem).NeoMl; 

describe('Test correct models', () => {
    test('Complete data block', async() => {
        const model = await assertModelNoErrors(`
        data myData {
            source = "C:/helloData"
            label = "myClassToPredict"
            drop = "unusedFeature1" "unusedFeature2"
            scaler = MinMax
        }
        `)
        const myData = model.all_data[0];
        expect(myData.name).toBe("myData");
        expect(myData.source).toBe("C:/helloData");
        expect(myData.drop).toStrictEqual(["unusedFeature1","unusedFeature2"]);
        expect(myData.scaler).toBe("MinMax");
    })

    test('All complete algo blocks', async() => {
        const model = await assertModelNoErrors(`
        algo mySvmModel svm {
            C = 0.0
            kernel = sigmoid
         }
 
        algo myKnnModel knn {
            n_neighbors = 8
            weights = distance
        }

        algo myDecisionTreeModel decisionTree {
            criterion = entropy
            splitter = random
            max_depth = 10
        }

        algo myMlpModel mlp {
            hidden_layer_sizes = 5
        }
        `)

        expect(model.all_algos[0].$type).toBe(SVM);
        if(isSVM(model.all_algos[0])) {
            const mySVM = model.all_algos[0];
            expect(mySVM.name).toBe("mySvmModel");
            expect(mySVM.C).toBe("0.0");
            expect(mySVM.kernel).toBe("sigmoid");
        }
        expect(model.all_algos[1].$type).toBe(KNN);
        if(isKNN(model.all_algos[1])) {
            const myKNN = model.all_algos[1];
            expect(myKNN.name).toBe("myKnnModel");
            expect(myKNN.n_neighbors).toBe(8);
            expect(myKNN.weights).toBe("distance");
        }
        expect(model.all_algos[2].$type).toBe(DecisionTree);
        if(isDecisionTree(model.all_algos[2])) {
            const myDecisionTree = model.all_algos[2];
            expect(myDecisionTree.name).toBe("myDecisionTreeModel");
            expect(myDecisionTree.criterion).toBe("entropy");
            expect(myDecisionTree.splitter).toBe("random");
            expect(myDecisionTree.max_depth).toBe(10);
        }
        expect(model.all_algos[3].$type).toBe(MLP);
        if(isMLP(model.all_algos[3])) {
            const myMLP = model.all_algos[3];
            expect(myMLP.name).toBe("myMlpModel");
            expect(myMLP.hidden_layer_sizes).toBe(5);
        }
    })

    test('Complete trainer block', async() => {
        const model = await assertModelNoErrors(`
        data myData {
            source = "C:/helloData"
            label = "myClassToPredict"
        }
         
        algo mySvmModel svm {
           C = 0.0
        }
         
        trainer {
            data = data.myData
            model = algo.mySvmModel
            train_test_split = 0.7
            show_metrics = false
        }
        `)

        const myTrainer = model.all_trainers[0];
        expect(myTrainer.data_ref.name).toBe("myData")
        expect(myTrainer.algo_ref.name).toBe("mySvmModel");
        expect(myTrainer.train_test_split).toBe("0.7");
        expect(myTrainer.show_metrics).toBe("false");
    })
});

describe('Test illegal models', () => {
    test('Non-unique names for data', async () => {
        await assertModelErrors(`
        data myData {
            source = "C:/helloData"
        }

        data myData {
            source = "C:/holaData"
        }
        `)
    });

    test('Non-unique names for algo', async () => {
        await assertModelErrors(`
        algo myFirstModel svm {
            C = 0.0
        }

        algo myFirstModel svm {
            C = 0.5
        }
        `)
    });

    test('Trainer references a non existing data block', async() => {
        await assertModelErrors(`
        data myData {
            source = "C:/helloData"
        }
         
        algo mySvmModel svm {
           C = 0.0
        }
         
        trainer {
            data = data.helloData
            model = algo.mySvmModel
        }
        `)
    });

    test('Trainer references a non existing algo block', async() => {
        await assertModelErrors(`
        data myData {
            source = "C:/helloData"
        }
         
        algo mySvmModel svm {
           C = 0.0
        }
         
        trainer {
            data = data.myData
            model = algo.myFirstModel
        }
        `)
    });
});


async function assertModelNoErrors(modelText: string) : Promise<Model> {
    var doc : LangiumDocument<AstNode> = await parseDocument(services, modelText)
    const db = services.shared.workspace.DocumentBuilder
    await db.build([doc], {validation: true});
    const model = (doc.parseResult.value as Model);
    expect(model.$document?.diagnostics?.length).toBe(0);
    return model;    
}

async function assertModelErrors(modelText: string) {
    var doc : LangiumDocument<AstNode> = await parseDocument(services, modelText)
    const db = services.shared.workspace.DocumentBuilder
    await db.build([doc], {validation: true});
    const model = (doc.parseResult.value as Model);
    expect(model.$document?.diagnostics?.length).toBeGreaterThan(0);  
}