grammar NeoML

entry ENTRY:
    (all_data+=Data | all_trainers+=Trainer | all_models+=Model)*;

Data:
    'data' name=ID "{" 
        "source" "=" source=STRING 
        ("label" "=" label=STRING )?
        ("drop" "=" drop+=STRING+ )?
        ("scaler" "=" scaler= SCALER )?
    "}";

DataRef:
    'data.'name=ID;

Model:
    SVM;

SVM:
    'model' name=ID 'svm' '{'
        ('C' '=' C=PERCENTAGE)?
        ('kernel' '=' kernel=("gamma" | "auto"))?

    '}';

ModelRef:
    'model.'name=ID;

Trainer:
    'trainer' '{'
        'data' '=' DataRef
        ('show_metrics' '=' show_metrics=("true" | "false"))?
        'model' '=' ModelRef
    '}';

hidden terminal WS: /\s+/;
terminal SCALER: /MinMax|Standard|AbsMax/;
terminal ID: /[a-zA-Z][\w]*/;
terminal STRING: /"[^"]*"|'[^']*'/;
terminal PERCENTAGE: /0.\d+|1/;

hidden terminal ML_COMMENT: /\/\*[\s\S]*?\*\//;
hidden terminal SL_COMMENT: /\/\/[^\n\r]*/;