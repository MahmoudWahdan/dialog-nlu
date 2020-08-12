# Dialog System NLU
Tensorflow and Keras Implementation of the state of the art researches in Dialog System NLU. 
Tested on Tensorflow version 2.x
Recently, using Huggingface Transformers library for better models coverage and other languages support.

You can still access the old version (TF 1.15.0) on TF_1 branch


## Implemented Papers
### NLU Papers
- [BERT for Joint Intent Classification and Slot Filling](https://arxiv.org/abs/1902.10909)
### Model Compression Papers
- [Poor Man’s BERT: Smaller and Faster Transformer Models](https://arxiv.org/abs/2004.03844)
### BERT / ALBERT for Joint Intent Classification and Slot Filling

![Joint BERT](img/joint_bert.PNG?)

### Poor Man’s BERT: Smaller and Faster Transformer Models

![Layer-dropping Strategies](img/Layer_dropping_Strategies.PNG)

#### Supported data format:
- Data format as in the paper `Slot-Gated Modeling for Joint Slot Filling and Intent Prediction` (Goo et al):
	- Consists of 3 files:
		- `seq.in` file contains text samples (utterances)
		- `seq.out` file contains tags corresponding to samples from `seq.in`
		- `label` file contains intent labels corresponding to samples from `seq.in`

#### Datasets included in the repo:
- Snips Dataset (`Snips voice platform: an embedded spoken language understanding system for private- by-design voice interfaces` )(Coucke et al., 2018), which is collected from the Snips personal voice assistant. 
	- The training, development and test sets contain 13,084, 700 and 700 utterances, respectively. 
	- There are 72 slot labels and 7 intent types for the training set.

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

```
python train_joint_bert.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_bert_model --epochs=5 --batch=64 --type=bert
```

```
python train_joint_bert.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_albert_model --epochs=5 --batch=64 --type=albert
```

```
python train_joint_bert_crf.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_bert_crf_model --epochs=5 --batch=32 --type=bert
```

```
python train_joint_bert_crf.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_albert_crf_model --epochs=5 --batch=32 --type=albert
```

**Example to do incremental training:**

```
python train_joint_bert.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_albert_model2 --epochs=5 --batch=64 --type=albert --model=saved_models/joint_albert_model
```


#### Evaluating the Joint BERT / ALBERT NLU model:
##### Required Parameters:
- ```--model``` or ```-m``` Path to joint BERT / ALBERT NLU model.
- ```--data``` or ```-d``` Path to data in Goo et al format.
##### Optional Parameters:
- ```--batch``` or ```-bs``` Batch size.
- ```--type``` or ```-tp``` to choose between `bert` and `albert`. Default is `bert`


```
python eval_joint_bert.py --model=saved_models/joint_bert_model --data=data/snips/test --batch=128 --type=bert
```

```
python eval_joint_bert.py --model=saved_models/joint_albert_model --data=data/snips/test --batch=128 --type=albert
```

```
python eval_joint_bert_crf.py --model=saved_models/joint_bert_crf_model --data=data/snips/test --batch=128 --type=bert
```

```
python eval_joint_bert_crf.py --model=saved_models/joint_albert_crf_model --data=data/snips/test --batch=128 --type=albert
```



### Integration with Huggingface Transformers library
[Huggingface Transformers](https://github.com/huggingface/transformers) has a lot of transformers-based models. The idea behind the integration is to be able to support more architectures as well as more languages.

Supported Models Architecture:
|Model|Pretrained Model Example|Layer Prunning Support|
|---|---|---|
|TFBertModel|```bert-base-uncased```|Yes|
|TFDistilBertModel|```distilbert-base-uncased```|Yes|
|TFAlbertModel|```albert-base-v1``` or ```albert-base-v2```|Not yet|
|TFRobertaModel|```roberta-base``` or ```distilroberta-base```|Not yet|
And more models integration to come


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

Using Transformers Bert example:
``` bash
python train_joint_trans.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_trans_model --epochs=3 --batch=64 --cache_dir=transformers_cache_dir  --trans=bert-base-uncased --from_pt=false
```

Using Transformers DistilBert example:
``` bash
python train_joint_trans.py --train=data/snips/train --val=data/snips/valid --save=saved_models/joint_distilbert_model --epochs=3 --batch=64 --cache_dir=transformers_cache_dir  --trans=distilbert-base-uncased --from_pt=false
```

#### Evaluating the Joint Transformer NLU model:
##### Required Parameters:
|Argument|Description|Is Required|Default|
|---|---|---|---|
|```--model``` or ```-m```|Path to joint Transformer NLU model.|Yes||
|```--data``` or ```-d```|Path to data in Goo et al format.|Yes||
|```--batch``` or ```-bs```|Batch size.|No|128|

Using Transformers Bert example:
``` bash
python eval_joint_trans.py --model=saved_models/joint_trans_model --data=data/snips/test --batch=128
```

Using Transformers DistilBert example:
``` bash
python eval_joint_trans.py --model=saved_models/joint_distilbert_model --data=data/snips/test --batch=128
```


#### Running a basic REST service for the Joint BERT / ALBERT NLU model:
##### Required Parameters:
- ```--model``` or ```-m``` Path to joint BERT / ALBERT NLU model.
##### Optional Parameters:
- ```--type``` or ```-tp``` to choose between `bert` and `albert`. Default is `bert`


```
python bert_nlu_basic_api.py --model=saved_models/joint_albert_model --type=albert
```

##### Sample request:
- POST
- Payload: 
```
{
	"utterance": "make me a reservation in south carolina"
}
```

##### Sample Response:
```
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



