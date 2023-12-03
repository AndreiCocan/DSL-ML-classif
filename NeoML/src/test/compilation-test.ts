import { describe, expect, test } from 'vitest';

import { type Model} from '../language/generated/ast.js';

import { AstNode, CompositeGeneratorNode, EmptyFileSystem, LangiumDocument } from 'langium';
import { parseDocument } from 'langium/test';
import { createNeoMlServices } from '../language/neo-ml-module.js';
import { generateClassifierPython } from '../cli/python-generator.js';

import {exec} from 'child_process';
import * as fs from 'fs';
import { generateClassifierR } from '../cli/r-generator.js';

const services = createNeoMlServices(EmptyFileSystem).NeoMl; 


describe.each([
  { neomlfile: '../Programs_examples/algo_block_decisionTree.neoml', quality:NaN, threshold: NaN },
  { neomlfile: '../Programs_examples/algo_block_knn.neoml', quality:NaN, threshold: NaN },
  { neomlfile: '../Programs_examples/algo_block_mlp.neoml', quality:NaN, threshold: NaN },
  { neomlfile: '../Programs_examples/algo_block_svm.neoml', quality:NaN, threshold: NaN },
  { neomlfile: '../Programs_examples/data_block_complete.neoml', quality:NaN, threshold: NaN },
  { neomlfile: '../Programs_examples/data_block_simple.neoml', quality:NaN, threshold: NaN },
  { neomlfile: '../Programs_examples/complete_minimalist_program.neoml', quality:1, threshold: 0.7 },
  // { neomlfile: '../Programs_examples/complete_program_train_2_models.neoml', quality:1, threshold: 0.7 },
   { neomlfile: '../Programs_examples/complete_program_with_unused_data_algo_blocks.neoml', quality:1, threshold: 0.7 },
   { neomlfile: '../Programs_examples/complete_program.neoml', quality:1, threshold: 0.7 },
   { neomlfile: '../Programs_examples/complete_test0.neoml', quality:1, threshold: 0.7 },
   { neomlfile: '../Programs_examples/complete_test1.neoml', quality:1, threshold: 0.7 },
   { neomlfile: '../Programs_examples/complete_test2.neoml', quality:0, threshold: 0.7 },
])('compilation and execution of $neomlfile', async ({ neomlfile, quality,threshold }) => {

    const neomlScript = fs.readFileSync(neomlfile, 'utf-8');

    const model_python = await assertModelNoErrors(neomlScript);

    const check_metrics:boolean=model_python.all_trainers.length>0 ? model_python.all_trainers.every(trainer =>trainer.show_metrics=='true'):false;

    
    const fileNode_python = new CompositeGeneratorNode();

    //python execution
    const generated_python_path =generateClassifierPython(model_python,'tmp.neoml','.',fileNode_python);

    const result_python=await new Promise<string>((resolve, reject) => {
    exec('python3 '+generated_python_path, (error, stdout, stderr) => {
        if (error) {
          console.error(`error: ${error.message}`);
        }
        else if (stderr) {
          console.warn(`stderr: ${stderr}`);
        }else{
            console.log(neomlfile,'execution (metrics',check_metrics,') stdout',stdout);
            resolve(stdout as string);
        }
        
      })
    });

  
    const model_r = await assertModelNoErrors(neomlScript);

    const fileNode_r = new CompositeGeneratorNode();

    const generated_r_path =generateClassifierR(model_r,'tmp.neoml','.',fileNode_r);

    const result_r=await new Promise<string>((resolve, reject) => {
    exec('Rscript '+generated_r_path, (error, stdout, stderr) => {
        if (error) {
          console.error(`error: ${error.message}`);
        }
        else if (stderr) {
          console.warn(`stderr: ${stderr}`);
        }else{
            console.log(neomlfile,'execution (metrics',check_metrics,') stdout',stdout);
            resolve(stdout as string);
        }
      })
    });

    if (check_metrics && !Number.isNaN(quality) && !Number.isNaN(threshold)){
      const python_accuracyValue=Number(result_python.split(":")[1]);

        console.log("test1",python_accuracyValue);
        test('checking python accuracy', () => { 
          if(quality){
            expect(threshold).toBeLessThan(python_accuracyValue);
          }else{
            expect(threshold).toBeGreaterThan(python_accuracyValue);
          }
  
        });
      
      const r_accuracyValue = Number(result_r.match(/accuracy: (\d+\.?\d*)/)?.[1]);

      console.log("test2",r_accuracyValue);
      test('checking r accuracy', () => {
        if (quality) {
          expect(threshold).toBeLessThan(r_accuracyValue);
        } else {
          expect(threshold).toBeGreaterThan(r_accuracyValue);
        }

      });

      console.log("test3",r_accuracyValue-python_accuracyValue);
      test('comparing python and r results', () => {
        expect(0.4).toBeGreaterThan(Math.abs(r_accuracyValue-python_accuracyValue));
      });

    }else{
      console.log("alternative r",result_r=='',"python",result_python=='');
      test('no output', () => {
        expect(result_r).toBe('');
        expect(result_python).toBe('');
      });

    }
});

async function assertModelNoErrors(modelText: string) : Promise<Model> {
    var doc : LangiumDocument<AstNode> = await parseDocument(services, modelText)
    const db = services.shared.workspace.DocumentBuilder
    await db.build([doc], {validation: true});
    const model = (doc.parseResult.value as Model);
    console.error(model.$document?.diagnostics);
    expect(model.$document?.diagnostics?.length).toBe(0);

    return model;    
}
