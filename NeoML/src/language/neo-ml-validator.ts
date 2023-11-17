import type { ValidationAcceptor, ValidationChecks } from 'langium';
import type { NeoMlAstType, Model } from './generated/ast.ts';
import type { NeoMlServices } from './neo-ml-module.ts';

/**
 * Register custom validation checks.
 */
export function registerValidationChecks(services: NeoMlServices) {
    const registry = services.validation.ValidationRegistry;
    const validator = services.validation.NeoMlValidator;
    const checks: ValidationChecks<NeoMlAstType> = {
        Model: [validator.checkUniqueDataNames, 
                validator.checkUniqueAlgoNames, 
                validator.checkTrainerReferencesExistingData,
                validator.checkTrainerReferencesExistingAlgos]
    };
    registry.register(checks, validator);
}

/**
 * Implementation of custom validations.
 */
export class NeoMlValidator {

    checkUniqueDataNames(model: Model, accept: ValidationAcceptor): void {
        const allDataNames = new Set();
        model.all_data.forEach(data => {
            if (allDataNames.has(data.name)) {
                accept('error',  `Data has non-unique name '${data.name}'.`,  {node: data, property: 'name'});
            }
            allDataNames.add(data.name);
        });
    }

    checkUniqueAlgoNames(model: Model, accept: ValidationAcceptor): void {
        const allAlgosNames = new Set();
        model.all_algos.forEach(algo => {
            if (allAlgosNames.has(algo.name)) {
                accept('error',  `Algo has non-unique name '${algo.name}'.`,  {node: algo, property: 'name'});
            }
            allAlgosNames.add(algo.name);
        });
    }

    checkTrainerReferencesExistingData(model: Model, accept: ValidationAcceptor): void {
        const allDataNames = new Set();
        model.all_data.forEach(data => {allDataNames.add(data.name);});
        model.all_trainers.forEach(trainer => {
            if(!allDataNames.has(trainer.data_ref.name)) {
                accept('error', `Trainer references a non existing data block '${trainer.data_ref.name}'`, 
                        {node: trainer, property: 'data_ref'});
            }
        });
    }

    checkTrainerReferencesExistingAlgos(model: Model, accept: ValidationAcceptor): void {
        const allAlgosNames = new Set();
        model.all_algos.forEach(algo => {allAlgosNames.add(algo.name);});
        model.all_trainers.forEach(trainer => {
            if(!allAlgosNames.has(trainer.algo_ref.name)) {
                accept('error', `Trainer references a non existing algo block '${trainer.algo_ref.name}'`,
                        {node: trainer, property: 'algo_ref'});
            }
        });
    }

}
