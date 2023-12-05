import { describe, expect , test} from 'vitest';

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
  { neomlfile: '../Programs_examples/algo_block_decisionTree.neoml' },
  { neomlfile: '../Programs_examples/algo_block_knn.neoml'},
  { neomlfile: '../Programs_examples/algo_block_mlp.neoml'},
  { neomlfile: '../Programs_examples/algo_block_svm.neoml'},
  { neomlfile: '../Programs_examples/data_block_complete.neoml'},
  { neomlfile: '../Programs_examples/data_block_simple.neoml' },
  { neomlfile: '../Programs_examples/complete_minimalist_program.neoml' },
  { neomlfile: '../Programs_examples/complete_program.neoml' },
  { neomlfile: '../Programs_examples/complete_program_with_unused_data_algo_blocks.neoml'},
  { neomlfile: '../Programs_examples/complete_program_train_2_models.neoml'},
  { neomlfile: '../Programs_examples/complete_test0.neoml'},
  { neomlfile: '../Programs_examples/complete_test1.neoml'},
  { neomlfile: '../Programs_examples/complete_test2.neoml'},
])('compilation and execution of $neomlfile', async ({ neomlfile}) => {


    const neomlScript = fs.readFileSync(neomlfile, 'utf-8');

    const model_python = await assertModelNoErrors(neomlScript);

    //const trainers_printing :boolean[]=model_python.all_trainers.length>0 ? model_python.all_trainers.map(trainer=>trainer.show_metrics=='true'):[];
    
    //const check_metrics:boolean=model_python.all_trainers.length>0 ? model_python.all_trainers.every(trainer =>trainer.show_metrics=='true'):false;

    // const data_ref_names=model_python.all_trainers.map(trainer=>trainer.data_ref.name)

    // const data_names=data_ref_names.map(data_ref_name=>model_python.all_data.filter(data=>data.name==data_ref_name).findLast())
    
    // model_python.all_trainers.map(trainer=>trainer.data_ref.name).map(name=>model_python.all_data.filter(data=>data.name==name))

    //const quality_list:boolean[] =model_python.all_data.filter(data=>model_python.all_trainers.map(trainer=>trainer.data_ref.name).includes(data.name)).map(data=>!data.source.includes("random"));
  
    const fileNode_python = new CompositeGeneratorNode();

    const classif_quality_list:boolean[]=model_python.all_trainers.map(
      trainer=>
        !model_python.all_data.filter(data=>data.name==trainer.data_ref.name)[0].source.includes("random")
    );
    const print_metrics_list:boolean[]=model_python.all_trainers.map(trainer=>trainer.show_metrics=='true');
    const print_index:number[]=print_metrics_list.map((bool, index) => bool ? index : -1).filter(index => index !== -1);
    const threshold=0.65;

    
  
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
            //console.log(neomlfile,'execution (metrics',print_metrics_list,') (quality',classif_quality_list,') stdout',stdout);
            console.log(neomlfile,'python execution stdout',stdout);
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
            //console.log(neomlfile,'execution (metrics',print_metrics_list,') (quality',classif_quality_list,') stdout',stdout);
            console.log(neomlfile,'r execution stdout',stdout);
            resolve(stdout as string);
        }
      })
    });

    if (print_metrics_list.length>0 && print_metrics_list.some(metric_print=>metric_print)){
      //si il y'a des trainers, et si un des trainers print bien un résultat

 // checking Python result
    const python_accuracyValues = result_python.match(/Accuracy score : (\d+\.?\d*)/g)?.map(match => Number(match.split(":")[1])) || [];

    //console.log("python accuracies", python_accuracyValues);
    test('checking python accuracy', () => {
      print_index.forEach((quality_index,value_index) => {
        if (classif_quality_list[quality_index]) {
          expect(threshold).toBeLessThan(python_accuracyValues[value_index]);
        } else {
          expect(threshold).toBeGreaterThan(python_accuracyValues[value_index]);
        }
      });
    });

// checking R result
    const r_accuracyValues = result_r.match(/accuracy: (\d+\.?\d*)/g)?.map(match => Number(match.split(":")[1])) || [];

    //console.log("r accuracies", r_accuracyValues);
    test('checking r accuracy', () => {
      print_index.forEach((quality_index,value_index) => {
        if (classif_quality_list[quality_index]) {
          expect(threshold).toBeLessThan(r_accuracyValues[value_index]);
        } else {
          expect(threshold).toBeGreaterThan(r_accuracyValues[value_index]);
        }
      });
    });

// comparing both results
    //console.log("comparing both",r_accuracyValues.length,python_accuracyValues.length);
    test('comparing python and r results', () => {
      expect(r_accuracyValues.length).toBe(python_accuracyValues.length);
      r_accuracyValues.forEach((r_accuracy,index)=>{
          expect(0.4).toBeGreaterThan(Math.abs(r_accuracy-python_accuracyValues[index]));          
      })
      
    });

    }else{
      //console.log("alternative r",result_r=='',"python",result_python=='');
      test('no output', () => {
        expect(result_r).toBe('');
        expect(result_python).toBe('');
      });

    }

    
}
);

async function assertModelNoErrors(modelText: string) : Promise<Model> {
    var doc : LangiumDocument<AstNode> = await parseDocument(services, modelText)
    const db = services.shared.workspace.DocumentBuilder
    await db.build([doc], {validation: true});
    const model = (doc.parseResult.value as Model);
    console.error(model.$document?.diagnostics);
    expect(model.$document?.diagnostics?.length).toBe(0);

    return model;    
}
