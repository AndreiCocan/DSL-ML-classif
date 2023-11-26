import { Model} from '../language/generated/ast.js';
import { CompositeGeneratorNode } from 'langium';
import { generateClassifierPython } from './python-generator.js';
import chalk from 'chalk';
import { generateClassifierR } from './r-generator.js';


export function generateClassifier(model: Model, filePath: string, destination: string | undefined, language: string | undefined): string {

    
    const fileNode = new CompositeGeneratorNode();

    switch(language) { 
        case "Python": { 
           return generateClassifierPython(model,filePath,destination,fileNode);
        } 
        case "R": { 
           return generateClassifierR(model,filePath,destination,fileNode); 
        } 
        default: { 
            console.log(chalk.redBright("Wrong language. It can be either Python or R."));
           break;
        } 
     }

    return "";
}


