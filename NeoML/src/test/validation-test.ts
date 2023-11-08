import { describe, expect, test } from 'vitest';

import type { Model } from '../language/generated/ast.js';

import { AstNode, EmptyFileSystem, LangiumDocument } from 'langium';
import { parseDocument } from 'langium/test';
import { createNeoMlServices } from '../language/neo-ml-module.js';

const services = createNeoMlServices(EmptyFileSystem).NeoMl; 

describe('Test correct models', () => {
    test('Single data block', async () => {
        const model = await assertModelNoErrors(`
        data myData {
            source = "C:/helloData"
        }
        `)
        expect(model.all_data[0].name).toBe("myData");
        expect(model.all_data[0].source).toBe("C:/helloData");
    });
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