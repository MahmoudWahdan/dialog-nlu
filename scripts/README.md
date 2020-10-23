## Here, we provide multiple scripts to train new model, incremental training, and evaluate model
#### Training the model with SICK-like data:
4 models implemented `joint_bert` and `joint_bert_crf` each supports `bert` and `albert`
##### Required Parameters:
- ```--train``` or ```-t``` Path to training data in Goo et al format.
- ```--val``` or ```-v``` Path to validation data in Goo et al format.
- ```--save``` or ```-s``` Folder path to save the trained model.
##### Optional Parameters:
- ```--epochs``` or ```-e``` Number of epochs.
- ```--batch``` or ```-bs``` Batch size.
- ```--type``` or ```-tp``` to choose between `bert` and `albert`. Default is `bert`
- ```--model``` or ```-m``` Path to joint BERT / ALBERT NLU model for incremental training.

**Using Tensorflow Hub Bert training example:**
```
python scripts/train_joint_bert.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_bert_model --epochs=5 --batch=64 --type=bert
```

**Using Tensorflow Hub Bert incremental training example:**
```
python scripts/train_joint_bert.py --train=data/snips/train --val=data/snips/valid --model=saved_models/joint_bert_model --save=saved_models/joint_bert_model2 --epochs=1 --batch=64
```

**Using Tensorflow Hub Albert training example:**
```
python scripts/train_joint_bert.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_albert_model --epochs=5 --batch=64 --type=albert
```

```
python train_joint_bert_crf.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_bert_crf_model --epochs=5 --batch=32 --type=bert
```

```
python train_joint_bert_crf.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_albert_crf_model --epochs=5 --batch=32 --type=albert
```




#### Training a joint Transformer model with SICK-like data:

##### Parameters:
|Argument|Description|Is Required|Default|
|---|---|---|---|
|```--train``` or ```-t```|Path to training data in Goo et al format.|Yes||
|```--val``` or ```-v```|Path to validation data in Goo et al format.|Yes||
|```--save``` or ```-s```|Folder path to save the trained model.|Yes||
|```--epochs``` or ```-e```|Number of epochs.|No|5|
|```--batch``` or ```-bs```|Batch size.|No|64|
|```--model``` or ```-m```|Path to joint trans NLU model for incremental training.|No||
|```--trans``` or ```tr```|Pretrained transformer model name or path. Is optional. Either --model OR --trans should be provided|No||
|```--from_pt``` or ```-pt```|Whether the --trans (if provided) is from pytorch or not|No|False|
|```--cache_dir``` or ```-c```|The cache_dir for transformers library. Is optional|No||

**Using Transformers Bert training example:**
``` bash
python scripts/train_joint_trans.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_trans_model --epochs=3 --batch=64 --cache_dir=transformers_cache_dir  --trans=bert-base-uncased --from_pt=false
```

**Using Transformers Bert incremental training example:**
``` bash
python scripts/train_joint_trans.py --train=data/snips/train --val=data/snips/valid --model=saved_models/joint_trans_model --save=saved_models/joint_trans_model2 --epochs=3 --batch=64
```

**Using Transformers DistilBert training example:**
``` bash
python scripts/train_joint_trans.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_distilbert_model --epochs=3 --batch=64 --cache_dir=transformers_cache_dir  --trans=distilbert-base-uncased --from_pt=false
```




#### Evaluating any NLU model:
We make use of [seqeval library](https://github.com/chakki-works/seqeval) for computing f1-score per tag level not per token level.
##### Required Parameters:
|Argument|Description|Is Required|Default|
|---|---|---|---|
|```--model``` or ```-m```|Path to joint Transformer NLU model.|Yes||
|```--data``` or ```-d```|Path to data in Goo et al format.|Yes||
|```--batch``` or ```-bs```|Batch size.|No|128|

**Using Transformers Bert example:**
``` bash
python scripts/evaluate.py --model=saved_models/joint_trans_model --data=data/snips/test --batch=128
```

**Using Transformers DistilBert example:**
``` bash
python scripts/evaluate.py --model=saved_models/joint_distilbert_model --data=data/snips/test --batch=128
```

**Using Tensorflow Hub Bert example:**
``` bash
python scripts/evaluate.py --model=saved_models/joint_bert_model --data=data/snips/test --batch=128
```






### Running a basic REST service for the Joint BERT / ALBERT NLU model:
##### Required Parameters:
- ```--model``` or ```-m``` Path to joint NLU model.
##### Optional Parameters:
- ```--port``` or ```-p``` port of the service. Default is `5000`


```
python scripts/nlu_basic_api.py --model=saved_models/joint_albert_model
```

```
python scripts/nlu_basic_api.py --model=saved_models/joint_distilbert_model
```


##### Sample request: 
- Method `/predict`
- POST
- Payload: 
```json
{
	"utterance": "make me a reservation in south carolina"
}
```

##### Sample Response:
```json
{
	"intent": {
		"confidence": "0.9888",
		"name": "BookRestaurant"
	}, 
	"slots": [
	{
		"slot": "state",
		"value": "south carolina",
		"start": 5,
		"end": 6
	}
	]
}
```