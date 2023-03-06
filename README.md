<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: End-To-End Named-Entity-Recognition for Cyber Threat Intelligence.

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

This example shows significant differences in expression of a labels meaning.  Here, files like "0b8806b38b52bebfe39ff585639e2ea2" or "Backdoor.AndroidOS.Chuli.a" (Indicator) are detected by "Kaspersky Lab" (Organization).  As this data do no cover relations between entities, the relation "detected by" is not of further interest.
# Evaluation of different NER-techniques
**Idea:** Compare the pipelines of CoreNLP and spaCy by focusing on their primary components. Hence, it might be interesting to see how they both work compared to each other. This means, both have several components leading to the final detection on named entities in texts. Another fascinating factor might be their implementation, usability in terms of programming effort and scalability.

**Possible Criteria:**


    1) General structure of pipelines
    2) Ease of use (Functionality)
    3) Changeability of components
    4) Domain adaptation
    5) Performance (Runtime, Scalability)

  | Tool                                                                                                                        | Basic Entities | BIO Format | Domain-Adaptation | Methods for Entity Recognition                        | Adding Pre-trained Models | End-to-End Readiness | Programming Language | Popularity on GitHub |
  |-----------------------------------------------------------------------------------------------------------------------------|----------------|------------|-------------------|-------------------------------------------------------|----------------------------|----------------------|----------------------|----------------------|
  | [spaCy](https://spacy.io/usage/linguistic-features#named-entities)                                                          | 18             | Yes        | Yes               | Ensemble, CNN, BILSTM, rule-based                     | Yes                        | Yes                  | Python               | 25,000+              |
  | [flairNLP](https://github.com/flairNLP/flair/blob/master/resources/docs/TUTORIAL_2_TAGGING.md#named-entity-recognition-ner) | 13             | Yes        | Yes               | Ensemble, CRF, BILSTM, rule-based                     | Yes                        | Yes                  | Python               | 12,000+              |
  | [NLTK](https://www.nltk.org/book/ch07.html#named-entity-recognition)                                                        | 5              | Yes        | Yes               | MaxEntropy, rule-based, regexp                        | No                         | No                   | Python               | 11,000+              |
  | [CoreNLP](https://stanfordnlp.github.io/CoreNLP/ner.html)                                                                   | 4              | Yes        | Yes               | Ensemble, CRF, rule-based, perceptron, neural network | No                         | No                   | Java                 | 8,000+               |

**spaCy:** spaCy has excellent documentation that is well-organized, comprehensive, and easy to follow. The documentation includes detailed guides for installation, usage, and customization, as well as a complete API reference. Additionally, spaCy has a vibrant community of developers who contribute to the documentation and provide support through forums and chat channels.

**flairNLP:** flairNLP also has good documentation, although it is not as extensive as spaCy's. The documentation includes guides for installation, usage, and customization, as well as examples and API reference.

**NLTK:** NLTK has been around for a long time and has a very extensive documentation, with comprehensive guides and tutorials for various natural language processing tasks. However, the documentation can be overwhelming for new users, as it covers a lot of ground and may require some programming experience to fully understand.

**CoreNLP:** CoreNLP has documentation that is adequate for basic usage, but it can be difficult to navigate and lacks examples and detailed explanations for more advanced features like adding new entities. Additionally, the documentation is less actively maintained than some other libraries, which may make it harder to get support when needed.
# spaCy:
The spaCy library provides a powerful and flexible pipeline for state-of-the-art natural language processing. At its core is the nlp object, which represents the pipeline itself. The pipeline is a sequence of tracable components that are applied to each input text in turn, with each component performing a specific task such as tokenization, part-of-speech tagging, or named entity recognition.
The nlp object is created by loading a pre-trained model, such as en_core_web_sm, which contains a set of pre-defined pipeline components for standard cases of NER. These components can be modified or extended as needed using the nlp.add_pipe() method. Each pipeline component takes a Doc object as input and returns a modified Doc object with additional annotations.
The Doc object represents a processed document of text, and contains a sequence of Token objects that represent individual words or other elements of the text, such as punctuation or whitespace. Each Token object has a variety of properties and annotations, such as its lemma, part-of-speech tag, and named entity label.
The nlp object also provides a range of convenient methods and attributes for working with processed documents, such as accessing specific tokens or entities, visualizing the document structure, or performing similarity calculations between documents.
<img src="https://spacy.io/images/pipeline.svg">

<img src="https://spacy.io/images/tok2vec-listener.svg">


## Basic Entities Labels
All spaCy pipelines provide a basic set of 18 entities to be interpreted as:

    CARDINAL : Numerals that do not fall under another type
    DATE : Absolute or relative dates or periods
    EVENT : Named hurricanes, battles, wars, sports events, etc.
    FAC : Buildings, airports, highways, bridges, etc.
    GPE : Countries, cities, states
    LANGUAGE : Any named language
    LAW : Named documents made into laws.
    LOC : Non-GPE locations, mountain ranges, bodies of water
    MONEY : Monetary values, including unit
    NORP : Nationalities or religious or political groups
    ORDINAL : "first", "second", etc.
    ORG : Companies, agencies, institutions, etc.
    PERCENT : Percentage, including "%"
    PERSON : People, including fictional
    PRODUCT : Objects, vehicles, foods, etc. (not services)
    QUANTITY : Measurements, as of weight or distance
    TIME : Times smaller than a day
    WORK_OF_ART : Titles of books, songs, etc.

These basic entity labels provide a solid ground for the most and common nlp setups. However, certain use-cases like NER-CTI other downstream-tasks require specific entity-labels like those of STIX. For these szenarios spaCy provides a detailed workflow.
## Workflow

<img src="https://spacy.io/images/projects.svg">

## Training

<img src="https://spacy.io/images/training.svg">


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
| `brace-yourself-data-is-coming` | Convert the CyNER data present in CoNLL format to .spacy format. |
| `initialize-configurations` | Initialize the training configurations for both models. |
| `debug-data-classic-model` | This method evaluates `train` and `dev` against the configuration of the classic model. |
| `train-classic-model` | This will run the training of the model as specified in `training_classic_model_config.cfg`. |
| `evaluate-classic-model` | This will evaluate `models/classic/model-best` with the held-out data for objectively testing the model performance. |
| `debug-data-foundation-model` | This method evaluates `train` and `dev` against the configuration of the foundation model. |
| `train-foundation-model` | This will run the training of the model as specified in `training_foundation_model_config.cfg`. Note: GPU with min. of 10GB memory required. |
| `evaluate-foundation-model` | This will evaluate `models/foundation/model-best` with the held-out data for objectively testing the model performance. |

### ⏭ Workflows

The following workflows are defined by the project. They
can be executed using [`spacy project run [name]`](https://spacy.io/api/cli#project-run)
and will run the specified commands in order. Commands are only re-run if their
inputs have changed.

| Workflow | Steps |
| --- | --- |
| `preparation` | `info` &rarr; `brace-yourself-data-is-coming` &rarr; `initialize-configurations` |
| `classic-model` | `debug-data-classic-model` &rarr; `train-classic-model` &rarr; `evaluate-classic-model` |
| `foundation-model` | `debug-data-foundation-model` &rarr; `train-foundation-model` &rarr; `evaluate-foundation-model` |

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