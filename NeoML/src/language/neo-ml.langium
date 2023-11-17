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
    SVM | KNN | DecisionTree | MLP;

SVM:
    'algo' name=ID 'svm' '{'
        ('C' '=' C=PERCENTAGE)?
        ('kernel' '=' kernel=("linear" | "poly" | "rbf" | "sigmoid" | "precomputed"))?
    '}';

KNN:
    'algo' name=ID 'knn' '{'
        ('n_neighbors' '=' n_neighbors=INT)?
        ('weights' '=' weights=("uniform" | "distance"))?
    '}';

DecisionTree:
    'algo' name=ID 'decisionTree' '{'
        ('criterion' '=' criterion=("gini" | "entropy" | "log_loss"))?
        ('splitter' '=' splitter=("best" | "random"))?
        ('max_depth' '=' max_depth=INT)?
    '}';

MLP:
    'algo' name=ID 'mlp' '{'
        ('hidden_layer_sizes' '=' hidden_layer_sizes+=INT+)?
    '}';

AlgoRef:
    'algo.'name=ID;

Trainer:
    'trainer' '{'
        'data' '=' data_ref=DataRef
        'model' '=' algo_ref=AlgoRef
        ('train_test_split' '=' train_test_split=PERCENTAGE)?
        ('show_metrics' '=' show_metrics=("true" | "false"))?
    '}';

hidden terminal WS: /\s+/;
terminal SCALER: /MinMax|Standard|AbsMax/;
terminal ID: /[a-zA-Z][\w]*/;
terminal STRING: /"[^"]*"|'[^']*'/;
terminal PERCENTAGE: /(0\.\d+)|(1\.0)/;
terminal INT: /[0-9]+/;

hidden terminal ML_COMMENT: /\/\*[\s\S]*?\*\//;
hidden terminal SL_COMMENT: /\/\/[^\n\r]*/;