import type { ValidationAcceptor, ValidationChecks } from 'langium';
import type { NeoMlAstType, Person } from './generated/ast.js';
import type { NeoMlServices } from './neo-ml-module.js';

/**
 * Register custom validation checks.
 */
export function registerValidationChecks(services: NeoMlServices) {
    const registry = services.validation.ValidationRegistry;
    const validator = services.validation.NeoMlValidator;
    const checks: ValidationChecks<NeoMlAstType> = {
        Person: validator.checkPersonStartsWithCapital
    };
    registry.register(checks, validator);
}

/**
 * Implementation of custom validations.
 */
export class NeoMlValidator {

    checkPersonStartsWithCapital(person: Person, accept: ValidationAcceptor): void {
        if (person.name) {
            const firstChar = person.name.substring(0, 1);
            if (firstChar.toUpperCase() !== firstChar) {
                accept('warning', 'Person name should start with a capital.', { node: person, property: 'name' });
            }
        }
    }

}
