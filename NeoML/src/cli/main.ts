import type { Model } from '../language/generated/ast.js';
import chalk from 'chalk';
import { Command } from 'commander';
import { NeoMLLanguageMetaData } from '../language/generated/module.js';
import { createNeoMlServices } from '../language/neo-ml-module.js';
import { extractAstNode } from './cli-util.js';
import { generateClassifier } from './generator.js';
import { NodeFileSystem } from 'langium/node';

export const generateAction = async (fileName: string, opts: GenerateOptions): Promise<void> => {
    const services = createNeoMlServices(NodeFileSystem).NeoMl;
    const model = await extractAstNode<Model>(fileName, services);
    const generatedFilePath = generateClassifier(model, fileName, opts.destination, opts.language);
    if(generatedFilePath != ""){
        console.log(chalk.green(`${opts.language} code generated successfully: ${generatedFilePath}`));
    }
        
};


export type GenerateOptions = {
    destination?: string;
    language?: string;
}

export default function main(): void {
    const program = new Command();

    program
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        //.version(require('../../package.json').version);

    const fileExtensions = NeoMLLanguageMetaData.fileExtensions.join(', ');
    program
        .command('generate')
        .argument('<file>', `source file (possible file extensions: ${fileExtensions})`)
        .option('-d, --destination <dir>', 'destination directory of generating')
        .option('-l, --language <lang>', "'R' or 'Python'")
        .description('generates JavaScript code that prints "Hello, {name}!" for each greeting in a source file')
        .action(generateAction);

    program.parse(process.argv);
}

main()