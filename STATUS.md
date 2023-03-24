It is our goal that our artifacts are *available*, *functional* and *reusable*. In the following, we describe how we aimed to fullfill the corresponding [requirements](https://www.acm.org/publications/policies/artifact-review-badging):

### :tada: Available
All our code and artifacts are archived on Zenodo under a permissive MIT license.
To further ease the use of the artifacts, the code and a pre-built docker container are distributed through github.

The case studies implemented are based on previously published, publicly available datasets.

### :tada::tada: Functional
We created dedictated scripts for different parts of our results, allowing to easily replicate our results.

We have verified the correct functioning and successful replication using our scripts on two dedicated machines.

### :tada::tada::tada: Reusable

We took a range of measures to facilitate the reusability of our artifacts, amongst which:

- A [REQUIREMENTS.md](REQUIRMENTS.md), [INSTALL.md](INSTALL.md) and [README.md](README.md) file provide detailed step-by-step guide on how to use our artifacts.
- Our code is structured into 15 modules of clearly separated functionality, thus clearly structured and making it easy to extract or extend parts of our experimentation.
- To facilitate subsequent research, all predictions, uncertainties and time measurements which form the foundation of our evaluation are released as CSV files, allowing other to quickly use this raw data to compare their appraoch against ours, without having to re-train, re-download or re-invoke any of the datasets and models we used.
- As a specicial case of the point mentioned before, we stored all our OpenAI API responses, allowing to re-use these predictions without having to pay for access to GPT-3 and without having a risk that due to internal changes at OpenAI, different responses might be received.
- We applied standard conventions regarding formatting and style of our code, thus facilitating editing by a third person.
- 3rd party artifacts we were not able to copy into our repository due to copyright or space reasons (e.g. datasets or model weights) are downloaded automatically, wherever possible.
