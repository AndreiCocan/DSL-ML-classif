import { describe, expect, test } from 'vitest';

import { type Model} from '../language/generated/ast.js';

import { AstNode, CompositeGeneratorNode, EmptyFileSystem, LangiumDocument } from 'langium';
import { parseDocument } from 'langium/test';
import { createNeoMlServices } from '../language/neo-ml-module.js';
import { generateClassifierPython } from '../cli/python-generator.js';

import {exec} from 'child_process';
import * as fs from 'fs';

const services = createNeoMlServices(EmptyFileSystem).NeoMl; 



describe.each([
  { neomlfile: '../Programs_examples/test0.neoml', quality:1, threshold: 0.8 },
  { neomlfile: '../Programs_examples/test1.neoml', quality:1, threshold: 0.8 },
  { neomlfile: '../Programs_examples/test2.neoml', quality:0, threshold: 0.79 },
])('compilation and execution of $neomlfile', async ({ neomlfile, quality,threshold }) => {

    const fileNode = new CompositeGeneratorNode();

    const neomlScript = fs.readFileSync(neomlfile, 'utf-8');

    const model = await assertModelNoErrors(neomlScript);

    const generated_path =generateClassifierPython(model,'tmp.neoml','.',fileNode);

    const result_python=await new Promise<string>((resolve, reject) => {
    exec('python3 '+generated_path, (error, stdout, stderr) => {
        if (error) {
          console.log(`error: ${error.message}`);
        }
        else if (stderr) {
          console.log(`stderr: ${stderr}`);
        }else{
            console.log(neomlfile,'execution','stdout',stdout);
            resolve(stdout as string);
        }
        
      })
    });

    test('checking accuracy', async() => {
      if(quality){
        expect(threshold).toBeLessThan(Number(result_python.split(":")[1]));
      }else{
        expect(threshold).toBeGreaterThan(Number(result_python.split(":")[1]));
      }

    })

});

async function assertModelNoErrors(modelText: string) : Promise<Model> {
    var doc : LangiumDocument<AstNode> = await parseDocument(services, modelText)
    const db = services.shared.workspace.DocumentBuilder
    await db.build([doc], {validation: true});
    const model = (doc.parseResult.value as Model);
    expect(model.$document?.diagnostics?.length).toBe(0);
    return model;    
}
