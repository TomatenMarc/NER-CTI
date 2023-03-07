<!-- SPACY PROJECT: AUTO-GENERATED DOCS START (do not remove) -->

# 🪐 spaCy Project: A End-To-End pipline for Named Entity Recognition of Cyber Threat Intelligence.


With the exponential growth of digital data, the need for efficient and effective processing of nearly unlimited information has become more pressing  than ever before. In the realm of [Cyber Threat Intelligence (CTI)](https://www.kaspersky.com/resource-center/definitions/threat-intelligence), one of the key challenges is  identifying the interesting parts of large volumes of fluent text data. Having stated this particular problem, [Named Entity Recognition (NER)](#task-definition-of-ner) is a crucial tool in this process, enabling analysts to  automatically identify and name entities such as persons, dates, organizations, and locations appearing in unstructured text.  However, [Named Entity Recognition of Cyber Threat Intelligence (NER-CTI)](#task-definition-of-ner-cti) differs from standard NER as it requires  domain adaptation of the labels, which may differ from the standard labels for entities in the real word.

While traditional NER tools can be complex and time-consuming, [spaCy](https://spacy.io) provides an end-to-end solution that offers fast and efficient  NER and other NLP tasks. One of the advantages of spaCy is its flexibility for domain adaptation, allowing users to [train their own models](https://spacy.io/usage/training) on custom labels and text data. This is a critical advantage in CTI, where the ability to adapt to new and evolving threats is essential.

In comparison, academic projects like [CoreNLP](https://stanfordnlp.github.io/CoreNLP/) may be less flexible when it comes to domain adaptation, as they are  often designed with a specific set of labels and data in mind. While these tools may be better suited for academic research and development, they may not be  as effective in real-world CTI scenarios where rapid adaptation is key.
CTI data generally comes from the [Cyberspace](https://www.oxfordlearnersdictionaries.com/definition/american_english/cyberspace) which is protected by organizations like the [Cyber Information Room (CIR)](https://www.bundeswehr.de/en/organization/the-cyber-and-information-domain-service). In the mindset of this report, parts of CIR are responsible for collecting, analyzing, and disseminating information related to cyber threats.

These information can take the form of digital reports from a variety of sources, including domain experts, cyber security facilities, new reports, and other digital sources. Processing these reports are essential in getting a global perspective of a situation, and this is where spaCy comes in. With its pre-trained models for [23 languages](https://spacy.io/usage/models#languages) including also the ability to process multi-lingual data,  analysts can easily start processing and analyzing reports without needing to invest time in building their own [NLP pipelines](https://spacy.io/api). Moreover, spaCy's focus on [deep learning](https://spacy.io/api/architectures#transformers) and [neural networks](https://spacy.io/usage/v2#summary) puts it  at the cutting edge of NLP research, offering the latest and most advanced techniques for NER and other NLP tasks.

One of the advantages of spaCy is its extensive [command line interface (CLI)](https://spacy.io/api/cli), which allows users to access a wide range of NLP  tasks without needing to write code. This CLI makes it particularly attractive for industry situations, where ease of use and deploy-ability are key concerns. In contrast, academic projects often require more customization and fine-tuning, which can make traditional NER tools more appropriate. However, for analysts working in the fast-paced world of [military intelligence (MI)](https://www.collinsdictionary.com/dictionary/english/military-intelligence), [spaCy's efficiency and ease of use](https://spacy.io/usage/facts-figures) make it an invaluable tool for processing and analyzing large volumes of CTI data.

Speaking generally about the domain of MI and CTI, the ability to quickly and accurately process large volumes of information is essential. By leveraging the power of spaCy, analysts can [streamline their workflow](https://spacy.io/usage/projects) and focus on analysis rather than technicalities. This report explores the exciting world of spaCy and how it is leading the charge in the fight against cyber attacks. 

With this horizon in mind, CIR is a critical part of MI operations, tasked with monitoring and responding to cyber threats.  By collecting and analyzing data from a variety of sources, the CIR provides a comprehensive view of the cyber threat landscape and enables analysts to take  proactive measures to defend against attacks.
Straightforward, this report focus on the cutting-edge technology spaCy and its potential use for military intelligence and its benefits for monitoring by CIR.

## :mortar_board: Task definition of NER
[Named Entity Recognition (NER)](https://deepai.org/machine-learning-glossary-and-terms/named-entity-recognition) is a fascinating field of  [Natural Language Processing (NLP)](https://www.ibm.com/topics/natural-language-processing) that involves the automatic identification and classification of  specific named entities such as people, places, organizations, and other objects of interest mentioned in text. This process involves sequence labeling at  the token level, where each word in a given text is analyzed for its context and syntax to determine whether it represents a named entity. NER can be performed using various techniques, including rule-based approaches, Conditional Random Fields (CRFs), and deep neural networks, each with its own advantages and limitations.
<object data="<iframe src="" width="100%" height="500px"></iframe>" type="application/pdf" width="100%" height="500px"> </object>

State-of-the-art NER systems, however, rely on advanced machine learning algorithms, such as Convolutional Neural Networks (CNNs),  Recurrent Neural Networks (RNNs), and Transformer models like BERT and GPT, to achieve high accuracy and efficiency.  Speaking about the performance, NER is often evaluated using standard benchmarks such as [CoNLL](#conll) datasets, which uses the [BIO](#bio) format to  annotate named entities in text. Another related dataset for benchmarking NER systems is [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19). 
### :green_apple: BIO
The [BIO](https://medium.com/analytics-vidhya/bio-tagged-text-to-original-text-99b05da6664) notation is a commonly used labeling scheme in NER tasks.  In this format, each token in a text is labeled with a prefix indicating whether it belongs to a named entity and, if so, what type of entity it is. The prefix is either "B", "I", or "O", where:
B (Beginning) indicates that the token is the beginning of a named entity. I (Inside) indicates that the token is inside a named entity. O (Outside) indicates that the token is not part of a named entity.
This is an example of how BIO might look in a sentence:

    John   lives in  New   York  City
    B-PER  O     O   B-LOC I-LOC I-LOC

In this example, "John" is the beginning of a person (PER) entity, "New York" is the beginning of a location (LOC) entity, and "City" is  inside the same location entity.
### :black_nib: CoNLL
The CoNLL format is a standard format for representing labeled sequences of tokens, often used for tasks like NER.  The format is named after the [Conference on Natural Language Learning (CoNLL)](https://www.conll.org/previous-tasks), which was first introduced  it in 2000 and since then proposes challenges whose solutions in turn define the state-of-the-art for NLP.
In the CoNLL format was introduced for the tasks of language-independent named entity recognition in [2002](https://www.clips.uantwerpen.be/conll2002/ner/)  and [2003](https://www.clips.uantwerpen.be/conll2003/ner/), and each line of a text file represents a single token and its associated label.  The first column contains the token itself, while subsequent columns contain labels for various linguistic features.  For example:

    John  B-PER
    lives O
    in  O
    New B-LOC
    York  I-LOC
    City  I-LOC

## :school_satchel: Task definition of NER-CTI
Named Entity Recognition with [Cyber Threat Intelligence (NER-CTI)](https://www.kaspersky.com/resource-center/definitions/threat-intelligence) is a  specialized application of Named Entity Recognition (NER) in the field of cybersecurity. NER-CTI involves the automatic identification and classification of specific named entities in cyber threat intelligence data, such as  indicators of compromise, malware names, IP addresses, domain names, and other cyber threat-related entities, often represented using  the [Structured Threat Information Expression (STIX)](#stix) format. The goal of NER-CTI is to extract actionable insights from large volumes of unstructured threat intelligence data to improve cybersecurity  defenses and response. NER-CTI techniques typically involve the use of machine learning algorithms, including deep  neural networks, to analyze and classify threat intelligence data. NER-CTI is an emerging area of research and has the potential to significantly enhance the capabilities of cybersecurity analysts in detecting, mitigating, and responding to cyber threats.
### :paperclip: STIX
As we do not deal with custom domain of named entity recognition, the the labels at hand differ from the standard entities like "GPE, ORG, LOC, PERCENT a.s.o.". The standard label-set for NER-CIT follows the definitions of [STIX](https://oasis-open.github.io/cti-documentation/stix/intro).
Having a closer look at STIX might also be interesting to find other CTI-datasets also including relationships. Additionally, STIX adds an interesting turn in working with CTI-data by introducing not only "Entities" and "Relations" but also "Sightings" defined as:  "belief that something in CTI (e.g., an indicator, malware, tool, threat actor, etc.) was seen".  This is especially fascinating, because a relation like "Kaspersky Lab detected Trojan.Win32.Agent" can be seen as the facts of having a CTI already broke the system. In contrast, a "sighting" is information streamed in real-time data not proven to be true or false, thus making the task of detecting cyberattacks  especially difficult.
## :floppy_disk: CyNER
CyNER is an open-source dataset for CTI and was introduced by [IBM T. J. Watson Research Center](https://arxiv.org/pdf/2204.05754.pdf). The CyNER dataset is a new dataset for NER-CTI, which consists of 4,372 cybersecurity-related sentences manually annotated with 6 types of entities, such as "Malware," "Vulnerability," and "Indicator." The dataset was created to address the lack of cybersecurity-specific NER datasets and to facilitate the development of more accurate and effective NER models for cybersecurity applications. In addition, the paper provides detailed statistics about the dataset, an analysis of its characteristics, and a baseline performance of 76.66% macro-F1  achieved with XML-RoBERTa-large.

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

  **Performance XLM-RoBERTa-large on test set:**

  | CyNER | Malware | Indicator   | System    | Organization   | Vulnerability    | 
  |-------|---------|-------------|-----------|----------------|------------------| 
  | Prec. | 79.82   | 78.34       | 70.36     | 70.64          | 100.0            | 
  | Rec.  | 75.11   | 86.62       | 79.93     | 60.16          | 80.00            | 
  | F1    | 77.39   | 82.27       | 74.84     | 64.98          | 88.89            | 

**Data-Format:** *CoNLL*

**Entity-Notation:** *(B I O)*

**Entity-Labels (STIX):** *'Organization', 'System', 'Malware', 'Indicator', 'O', 'Vulnerability'*

**Repository:** https://github.com/aiforsec/CyNER

**Example:**

    This malicious APK is 334326 bytes file , MD5 :
    O    O         O   O  O      O     O    O O   O

    0b8806b38b52bebfe39ff585639e2ea2 and is detected
    B-Indicator                      O   O  O

    by Kaspersky      Lab products   as " Backdoor.AndroidOS.Chuli.a " .
    O  B-Organization I-Organization O  O B-Indicator                O O

This example shows significant differences in expression of a labels meaning.  Here, files like "0b8806b38b52bebfe39ff585639e2ea2" or "Backdoor.AndroidOS.Chuli.a" (Indicator) are detected by "Kaspersky Lab" (Organization).  As this data do no cover relations between entities, the relation "detected by" is not of further interest.
# :bookmark_tabs: spaCy's text processing pipeline:
The SpaCy pipeline is a sequence of NLP components that are applied to a text document in order to extract meaning and structure from the text, including a  [tokenizer](https://spacy.io/api/tokenizer), a [part-of-speech tagger (POS)](https://spacy.io/api/tagger), a [dependency parser](https://spacy.io/api/dependencyparser), [NER](https://spacy.io/api/entityrecognizer), and all follow-up tasks like [text categorization](https://spacy.io/api/textcategorizer).
<figure>
  <img src="https://spacy.io/images/pipeline.svg" alt="Spacy pipeline diagram">
  <figcaption>A diagram of the Spacy pipeline <em>(Source: https://spacy.io/images/pipeline.svg)</em></figcaption>
</figure>

At its core is the [nlp](https://spacy.io/usage/processing-pipelines) object, which represents the pipeline itself.
The pipeline starts by tokenizing the text, or breaking it up into individual words or sub-word units. Then, the POS tagger assigns a part-of-speech tag to  each token, such as "noun", "verb", "adjective", etc. Next, the dependency parser analyzes the relationships between the tokens in the sentence, identifying  the grammatical structure and dependencies between words. Consequently, the NER component then identifies and extracts named entities, such as people, organizations, and locations, from the text.  This capability in turn can be used to extract structured data from unstructured text, such as building a knowledge graph or generating a database.

Also it's worth noting that the components of the SpaCy pipeline is designed to be flexible and modular, so developers can choose to use all or some of them, or  even add [custom components](https://spacy.io/usage/training#custom-functions) as needed.

Overall, a pipeline returns a so-called [Doc](https://spacy.io/api/doc#_title) object representing a processed document of text, that contains a sequence of  [Token](https://spacy.io/api/token) objects represent individual words or other elements of the text, such as punctuation or whitespace.  Each Token object has a variety of properties and annotations, such as its lemma, part-of-speech tag, and named entity labels. The nlp object also provides a range of convenient methods and attributes for working with processed documents, such as accessing specific tokens or entities, visualizing the document structure, or performing similarity calculations between documents.
## :telescope: Turning text into features
spacy's pipeline components, such as the part-of-speech tagger, dependency parser, and named entity recognizer, can listen to the text representation  components tok2vec or transformers to extract useful features for their respective tasks. These features are learned through pre-training on large corpora of  text, and can capture important semantic and syntactic relationships between words. By using pre-trained vectors, the text representation component can improve the accuracy and robustness of these downstream components, as they can better capture the nuances of natural language.
<figure>
  <img src="https://spacy.io/images/tok2vec-listener.svg" alt="Spacy listener diagram">
  <figcaption>A diagram of how spacy's components listen to text features. <em>(Source: https://spacy.io/images/tok2vec-listener.svg)
</em></figcaption> </figure>

The [tok2vec](https://spacy.io/api/tok2vec#_title) component in spacy's pipeline uses a [convolutional neural network (CNN)](https://spacy.io/models#design-cnn) architecture to generate word embeddings that capture local context information. These embeddings are pre-trained on a large corpus of text using a  self-supervised learning approach. Specifically, the CNN is trained to predict the word that appears in the center of a fixed-size window of surrounding words. By doing so, the model learns to capture important local context information that is useful for downstream tasks such as part-of-speech tagging and named  entity recognition.

The [transformers](https://spacy.io/usage/embeddings-transformers) component, on the other hand, uses a  [self-attention mechanism](https://machinelearningmastery.com/the-transformer-attention-mechanism/) to generate contextualized word embeddings that  capture both local and global context information. The model is pre-trained on large corpora of text using a [self-supervised learning](https://ai.facebook.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/) approach known as masked language modeling. Specifically, the model is trained to predict the masked words in a sentence by attending to the surrounding words in the context.  This approach allows the model to learn rich contextual representations that capture complex relationships between words and are useful for a wide range of natural language processing tasks.

When encountering out-of-distribution words, tok2vec and transformers handle them in different ways. For tok2vec, these words are treated as unknown and  assigned a zero vector. However, this can sometimes lead to poor performance on tasks that require handling of rare or unseen words. To address this, spacy  provides options to fine-tune the tok2vec component on domain-specific data to better handle out-of-distribution words. Transformers, on the other hand, use a contextualized approach that allows the model to generate a representation for any word in the context of  the surrounding text. This means that even if a word is out of distribution, the transformer can still  [generate a meaningful representation](https://ai.googleblog.com/2021/12/a-fast-wordpiece-tokenization-system.html) based on the context in which the word appears.

--------------

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

## Hardware

    Environment: Google-Colab  
    GPU 0: Tesla T4 (UUID: GPU-616858e9-5652-63ea-cf1a-e7df8574dc8e)
    Used: ~13.3 of 16 GB GPU for training over 1h and 40 minutes.
    
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
    | N/A   44C    P0    24W /  70W |      0MiB / 15360MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+

    Architecture:                    x86_64
    CPU op-mode(s):                  32-bit, 64-bit
    Byte Order:                      Little Endian
    Address sizes:                   46 bits physical, 48 bits virtual
    CPU(s):                          2
    On-line CPU(s) list:             0,1
    Thread(s) per core:              2
    Core(s) per socket:              1
    Socket(s):                       1
    NUMA node(s):                    1
    Vendor ID:                       GenuineIntel
    CPU family:                      6
    Model:                           85
    Model name:                      Intel(R) Xeon(R) CPU @ 2.00GHz
    Stepping:                        3
    CPU MHz:                         2000.212
    BogoMIPS:                        4000.42
    Hypervisor vendor:               KVM
    Virtualization type:             full
    L1d cache:                       32 KiB
    L1i cache:                       32 KiB
    L2 cache:                        1 MiB
    L3 cache:                        38.5 MiB
    NUMA node0 CPU(s):               0,1
    Vulnerability Itlb multihit:     Not affected
    Vulnerability L1tf:              Mitigation; PTE Inversion
    Vulnerability Mds:               Vulnerable; SMT Host state unknown
    Vulnerability Meltdown:          Vulnerable
    Vulnerability Mmio stale data:   Vulnerable
    Vulnerability Retbleed:          Vulnerable
    Vulnerability Spec store bypass: Vulnerable
    Vulnerability Spectre v1:        Vulnerable: __user pointer sanitization and use
                                     rcopy barriers only; no swapgs barriers
    Vulnerability Spectre v2:        Vulnerable, IBPB: disabled, STIBP: disabled, PB
                                     RSB-eIBRS: Not affected
    Vulnerability Srbds:             Not affected
    Vulnerability Tsx async abort:   Vulnerable
    Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtr
                                     r pge mca cmov pat pse36 clflush mmx fxsr sse s
                                     se2 ss ht syscall nx pdpe1gb rdtscp lm constant
                                     _tsc rep_good nopl xtopology nonstop_tsc cpuid 
                                     tsc_known_freq pni pclmulqdq ssse3 fma cx16 pci
                                     d sse4_1 sse4_2 x2apic movbe popcnt aes xsave a
                                     vx f16c rdrand hypervisor lahf_lm abm 3dnowpref
                                     etch invpcid_single ssbd ibrs ibpb stibp fsgsba
                                     se tsc_adjust bmi1 hle avx2 smep bmi2 erms invp
                                     cid rtm mpx avx512f avx512dq rdseed adx smap cl
                                     flushopt clwb avx512cd avx512bw avx512vl xsaveo
                                     pt xsavec xgetbv1 xsaves arat md_clear arch_cap
                                     abilities


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