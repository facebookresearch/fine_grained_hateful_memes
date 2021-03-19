# WOAH Shared Task Fine Grained Hateful Memes Classification

## Background

In order for AI to become a more effective tool for detecting hate speech, it must be able to understand content the way people do: holistically. When viewing a meme, for example, we don’t think about the words and photo independently of each other; we understand the combined meaning together. This is extremely challenging for machines, however, because it means they can’t just analyze the text and the image separately. They must combine these different modalities and understand how the meaning changes when they are presented together. To accelerate research on multimodal understanding and hate speech, Facebook AI created the hateful memes challenge, and released a dataset containing 10,000+ new multimodal examples.

Hate speech, however, continues to be an important challenge, and multimodal hate speech remains an especially difficult machine learning problem. Hate speech is defined as a direct attack (characterized as violent or dehumanizing speech, harmful stereotypes, statements of inferiority, expressions of contempt, disgust or dismissal, cursing, and calls for exclusion or segregation) against people on the basis of what we call protected characteristics (characterized as race, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity and serious disease). We operationalize this definition by making fine-grained labels for protected classes and attack types available as additional annotations on the hateful memes dataset. 



## Shared Task Description
We present two detection tasks on the newly annotated hateful memes dataset: 
Task A (multi-label): For each meme, detect the protected category. Protected categories are: race, disability, religion, nationality, sex. If the meme is not_hateful the protected category is: pc_empty.
Task B (multi-label): For each meme, detect the attack type. Attack types are: contempt, mocking, inferiority, slurs, exclusion, dehumanizing, inciting_violence. If the meme is not_hateful the protected category is: attack_empty.

Tasks A and B are multi-label because memes can contain attacks against multiple protected categories and can involve multiple attack types. 

### Input Data Format
The fine-grained hateful memes task consists of predicting 3 classes - hate detection (hateful or not), protected classes, and attacks.

The above meme is represented in the JSON record below. The various fields defined are
| Field | Description |
| -----  | ---------- |
| img | relative path of the raw image |
| text | extracted meme text |
| set_name | data partition - training and development splits |
| pc | Protected category annotations, includes annotations from up to 3 annotators |
| gold_pc | reference labels for protected categories (based n majority voting) | 
| attacks | Attack type annotations, includes annotations  from up to 3 annotators |
| gold_attack | Reference labels for attack types (based on majority voting) |
| gold_hate | Reference labels for hate classification task |
| id | Unique identifier of the record |

```
{
  "img": "img/83745.png",
  "text": "it is time.. to send these parasites back to the desert",
  "set_name": "dev_seen",
  "pc": [
    [
      "race"
    ],
    [
      "nationality",
      "religion"
    ]
  ],
  "attack": [
    [
      "dehumanizing"
    ],
    [
      "inciting_violence",
      "dehumanizing",
      "exclusion"
    ]
  ],
  "gold_pc": [
    "race"
  ],
  "gold_attack": [
    "dehumanizing"
  ],
  "id": 83745,
  "gold_hate": [
    "hateful"
  ]
}
```

### Submission Format
The predictions need to be in JSON format using the field descriptors below
| Field | Description |
| -----  | ---------- |
| pred_pc | Dictionary of {label:score} for the hate classification task | 
| pred_attack | Dictionary of {label:score} for the attack category task |
| pred_hate | Dictionary of {label:score} for the hate classification task |
| id | Unique identifier of the record, should match the input dataset ids |

```
{
  "id": 83745,
  "set_name": "dev_seen",
  "pred_hate": {
    "not_hateful": 0.9999529123,
    "hateful": 4.55694e-05
  },
  "pred_pc": {
    "disability": 2.56113e-05,
    "nationality": 2.19724e-05,
    "pc_empty": 0.9999549389,
    "race": 2.94644e-05,
    "religion": 2.72773e-05,
    "sex": 1.61978e-05
  },
  "pred_attack": {
    "attack_empty": 0.9999647141,
    "contempt": 2.98673e-05,
    "dehumanizing": 4.10547e-05,
    "exclusion": 2.69093e-05,
    "inciting_violence": 4.66734e-05,
    "inferiority": 2.6985e-05,
    "mocking": 2.68056e-05,
    "slurs": 2.02888e-05
  }
}
```

## Evaluation

We use the standard roc_auc metric provided in sklearn library to score each of the 3 tasks - hate, pc and attack. Here's an example of how to use the script, and a sample output
```
Input: python scripts/scorer.py -g data/annotations/dev_seen.json -p data/baselines/independent/predictions/dev_seen.json

Output:
{'task': 'hate', 'f1_micro': 0.6468253968253969, 'roc_auc': 0.7031958616780045}
{'task': 'pc', 'f1_micro': 0.6209523809523809, 'roc_auc': 0.8511728239769002}
{'task': 'attack', 'f1_micro': 0.61284046692607, 'roc_auc': 0.870857207208931}
```

## How to Participate
* Join woah2021task@googlegroups.com. Your request should include the first name, last name, and affiliation of all team members.
* Get the original phase 1 hateful memes dataset [here](https://www.drivendata.org/competitions/64/hateful-memes

## Rules
* You must submit your code with your predictions, and make it available open source. 
* You cannot hand label any of the entries or manually assign them scores. 
* You should treat the test set examples as independent.
* Your system should predict protected category and attacks over the entire dataset, for non-hateful the model should be able to predict `pc_emtpy` and `attack_emtpy`
* If you do not adhere to the spirit of the competition rules then your entry will be rejected.




## Shared Task Organizers
* Shaoliang Nie, Facebook AI
* Aida Davani, University of Southern California.
* Lambert Mathias, Facebook.
* Douwe Kiela, Facebook.
* Zeerak Waseem, University of Sheffield
* Bertie Vidgen, Alan Turing Institute
* Vinodkumar Prabhakaran, Google Research

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.

## License
See the [LICENSE](LICENSE)
