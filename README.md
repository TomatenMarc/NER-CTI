<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: End-To-End Named-Entity-Recognition for Cyber Threat Intelligence.

This has to be added.
## Approach
The approach of this report is twofold. In the first instance, the traditional approaches are contrasted. Therefore, the [CoreNLP pipeline](https://stanfordnlp.github.io/CoreNLP/ner.html#additional-tokensregexner-rules), which determines the entities in last instance on the basis [Conditional Random Field (CRF)](https://towardsdatascience.com/conditional-random-fields-explained-e5b8256da776),  and the [extended spaCy pipeline](https://spacy.io/usage/processing-pipelines) are compared with one another.  Extended means that spaCy is ythe modern one of both libraries and makes it possible for example to replace individual components such as the feature  extraction by means of different embedding techniques. Based on this, a [foundation model](https://research.ibm.com/blog/what-are-foundation-models), i.e. a  more specialized [transformer pipeline](https://spacy.io/usage/v3#features-transformers-pipelines), is to be integrated into this process and evaluated.
## Task definition of NER
NER stands for Named-Entity-Recognition, which is a subtask of Natural Language Processing (NLP) that involves identifying and categorizing named entities in text into predefined categories such as person names, organization names, locations, and others.
## Task definition of NER-CTI
NER-CTI stands for Named-Entity-Recognition for Cyber-Threat-Intelligence, which is a subtask of NER that involves identifying and categorizing named entities related to Cyber-Threats in text into predefined categories such as IPs, URLs, protocols, locations or threat participants.
## BIO notation
The BIO notation is a commonly used labeling scheme in NER tasks. In this format, each token in a text is labeled with a prefix indicating whether it belongs  to a named entity and, if so, what type of entity it is. The prefix is either "B", "I", or "O", where:
B (Beginning) indicates that the token is the beginning of a named entity. I (Inside) indicates that the token is inside a named entity. O (Outside) indicates that the token is not part of a named entity.
This is an example of how BIO might look in a sentence:

    John   lives in  New   York  City
    B-PER  O     O   B-LOC I-LOC I-LOC

In this example, "John" is the beginning of a person (PER) entity, "New York" is the beginning of a location (LOC) entity, and "City" is  inside the same location entity.
## CoNLL format
The CoNLL format is a standard format for representing labeled sequences of tokens, often used for tasks like named entity recognition (NER) or part-of-speech (POS) tagging. The format is named after the [Conference on Natural Language Learning (CoNLL)](https://www.conll.org/previous-tasks), which first introduced  it in 2000.
In the CoNLL format was introduced for the tasks of language-independent named entity recognition in [2002](https://www.clips.uantwerpen.be/conll2002/ner/)  and [2003](https://www.clips.uantwerpen.be/conll2003/ner/), each line of a text file represents a single token and its associated labels.  The first column contains the token itself, while subsequent columns contain labels for various linguistic features.  For example, in a typical NER task, the second column might contain the named entity label for each token, while in a POS tagging task, it might contain  the part-of-speech tag.

    John  B-PER
    loves O
    in  O
    New B-LOC
    York  I-LOC
    City  I-LOC

## About a universal annotation language for CTI data (STIX)
As we do not deal with custom domain of named entity recognition, the the labels at hand differ from the standard entities like "GPE, ORG, LOC, PERCENT a.s.o.". The standard label-set for NER-CIT follows the definitions of [STIX](https://oasis-open.github.io/cti-documentation/stix/intro).
Having a closer look at STIX might also be interesting to find other CTI-datasets also including relationships. Additionally, STIX adds an interesting turn in working with CTI-data by introducing not only "Entities" and "Relations" but also "Sightings" defined as:  "belief that something in CTI (e.g., an indicator, malware, tool, threat actor, etc.) was seen".  This is especially fascinating, because a relation like "Kaspersky Lab detected Trojan.Win32.Agent" can be seen as the facts of having a CTI already broke the system. In contrast, a "sighting" is information streamed in real-time data not proven to be true or false, thus making the task of detecting cyberattacks  especially difficult.
## CyNER
CyNER is an open-source dataset for CTI and was introduced by [IBM T. J. Watson Research Center](https://arxiv.org/pdf/2204.05754.pdf).

  **Token Distribution:**
  
  | CyNER | Token | Unique   | Documents |
  |-------|--------|---------|-----------|
  | Train | 68.191 | 7.954   | 2.811     |
  | Dev   | 19.530 | 3.370   | 813       |
  | Test  | 19.270 | 3.602   | 748       |

  **Label Distribution:**
  
  | CyNER | Malware | Indicator   | System    | Organization   | Vulnerability    | 
  |-------|---------|-------------|-----------|----------------|------------------| 
  | Train | 705     | 1.252       | 838       | 288            | 48               | 
  | Dev   | 254     |   247       | 182       |  92            |  9               | 
  | Test  | 242     |   301       | 249       | 134            | 10               | 

**Data-Format:** *CoNLL*

**Entity-Notation:** *(B I O)*

**Entity-Labels:** *'Organization', 'System', 'Malware', 'Indicator', 'O', 'Vulnerability'*

**Repository:** https://github.com/aiforsec/CyNER

**Example:**

    This malicious APK is 334326 bytes file , MD5 :
    O    O         O   O  O      O     O    O O   O

    0b8806b38b52bebfe39ff585639e2ea2 and is detected
    B-Indicator                      O   O  O

    by Kaspersky      Lab products   as " Backdoor.AndroidOS.Chuli.a " .
    O  B-Organization I-Organization O  O B-Indicator                O O

This example shows significant differences in expression of a labels meaning.  Here, files like "0b8806b38b52bebfe39ff585639e2ea2" or Backdoor.AndroidOS.Chuli.a" (Indicator) are detected by "Kaspersky Lab" (Organization).  As this data do no cover relations between entities, the relation "detected by" is not of further interest.


## 📋 project.yml

The [`project.yml`](project.yml) defines the data assets required by the
project, as well as the available commands and workflows. For details, see the
[spaCy projects documentation](https://spacy.io/usage/projects).

### ⏯ Commands

The following commands are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run).
Commands are only re-run if their inputs have changed.

| Command | Description |
| --- | --- |
| `info` | Creates the README.md of this project. |
| `convert` | Convert the CyNER data present in CoNLL format to .spacy format. |
| `initialize` | Initialize the configuration for training. |
| `debug-data` | This method adds `train.spacy` and `val.spacy` to `[paths]` section in `training_config.cfg` and evaluates the data against the model before training.. |
| `train` | This will run the training of the model as specified in `training_config.cfg`. |
| `evaluate` | This will evaluate `models/model-best` with the held-out data for objectively testing the model performance. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `all` | `info` &rarr; `convert` &rarr; `initialize` &rarr; `debug-data` &rarr; `train` &rarr; `evaluate` |

### 🗂 Assets

The following assets are defined by the project. They can
be fetched by running [`spacy project assets`](https://spacy.io/api/cli#project-assets)
in the project directory.

| File | Source | Description |
| --- | --- | --- |
| `assets/CyNER/train.txt` | URL | The training data of CyNER. |
| `assets/CyNER/valid.txt` | URL | The valid data of CyNER. This will be renamed to `dev` after calling `convert`. |
| `assets/CyNER/test.txt` | URL | The test data of CyNER. |

<!-- SPACY PROJECT: AUTO-GENERATED DOCS END (do not remove) -->