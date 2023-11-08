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
        Model: [validator.checkUniqueDataNames, validator.checkUniqueAlgoNames]
    };
    registry.register(checks, validator);
}

/**
 * Implementation of custom validations.
 */
export class NeoMlValidator {

    checkUniqueDataNames(model: Model, accept: ValidationAcceptor): void {
        // create a set of visited data blocks
        // and report an error when we see one we've already seen
        const reported = new Set();
        model.all_data.forEach(data => {
            if (reported.has(data.name)) {
                accept('error',  `Data has non-unique name '${data.name}'.`,  {node: data, property: 'name'});
            }
            reported.add(data.name);
        });
    }

    checkUniqueAlgoNames(model: Model, accept: ValidationAcceptor): void {
        // create a set of visited algo blocks
        // and report an error when we see one we've already seen
        const reported = new Set();
        model.all_algos.forEach(algo => {
            if (reported.has(algo.name)) {
                accept('error',  `Algo has non-unique name '${algo.name}'.`,  {node: algo, property: 'name'});
            }
            reported.add(algo.name);
        });
    }

}
