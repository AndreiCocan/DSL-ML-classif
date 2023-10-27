grammar NeoML

entry Model:
    (all_data+=Data | all_trainers+=Trainer | all_algos+=Algo)*;

Data:
    'data' name=ID "{" 
        "source" "=" source=STRING 
        ("label" "=" label=STRING )?
        ("drop" "=" drop+=STRING+ )?
        ("scaler" "=" scaler= SCALER )?
    "}";

DataRef:
    'data.'name=ID;

Algo:
    SVM;

SVM:
    'algo' name=ID 'svm' '{'
        ('C' '=' C=PERCENTAGE)?
        ('kernel' '=' kernel=("linear" | "poly" | "rbf" | "sigmoid" | "precomputed"))?

    '}';

AlgoRef:
    'algo.'name=ID;

Trainer:
    'trainer' '{'
        'data' '=' data_ref=DataRef
        'model' '=' algo_ref=AlgoRef
        'train_test_split' '=' train_test_split=PERCENTAGE?
        ('show_metrics' '=' show_metrics=("true" | "false"))?
    '}';

hidden terminal WS: /\s+/;
terminal SCALER: /MinMax|Standard|AbsMax/;
terminal ID: /[a-zA-Z][\w]*/;
terminal STRING: /"[^"]*"|'[^']*'/;
terminal PERCENTAGE: /0.\d+|1/;

hidden terminal ML_COMMENT: /\/\*[\s\S]*?\*\//;
hidden terminal SL_COMMENT: /\/\/[^\n\r]*/;