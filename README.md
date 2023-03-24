# Replication Package: Adopting Two Supervisors for Efficient Use of Large-Scale Remote Deep Neural Networks


## Installation

1. For hardware and software requirements, please refer to [REQUIREMENTS.md](REQUIREMENTS.md).
2. Then, to build and run the docker image, please refer to [INSTALL.md](INSTALL.md).

Once the docker container is running and the `/generated` folder is mounted as described in [INSTALL.md](INSTALL.md), you are ready to re-run our evaluation.

## How to re-run our evaluation
Our replication package exposes 4 scripts, each responsible for parts of the results shown in our paper. 

### :heavy_check_mark: Prerequisite Check: Static model performances (Parts of Table 1)

```
python /src/evaluation/a_model_performance.py
```

The script will run and write various things to the console output, ending with:

```
imdb values. Local only: 0.79432, Remote only 0.89476 
Issues values. Local only: 0.71051, Remote only 0.8232 
ImageNet values. Local only: 0.6783, Remote only 0.85232 
SQuADv2 (possible only) values. Local only: 0.63209, Remote only 0.69496 
SQuADv2 (all) values. Local only: 0.27988, Remote only 0.30771 
```

You may now verify that these numbers match the ones reported in Table 1. Note that the values for `SQuADv2 (possible only)` are not shown in Table 1 but calculated in the replication script for completeness.


### :heavy_check_mark: RQ1: Request-Accuracy (RAC) Plots (Figures 2 to 5)
Next, we will replicate the results for RQ1, i.e., Figures 2 to 5. If you want, feel free to delete the contents of the folder `generated/results/rac`, where these Figures are located. Then, run the replication script as follows:

```
python /src/evaluation/b_rq1_rac.py
```

After the script finished running, verify the content of the `generated/results/rac` to see the replicated plots. 


### :heavy_check_mark: RQ2: Supervised Setting Tables (Tables 2 to 6)
Next, we will replicate the results for RQ2, i.e., Tables 2 to 6. If you want, feel free to delete the contents of the folder `generated/results/rq2_tex_tables` and the file `generated/results/rq2_table.csv`. The former contains the latex tables used in our paper, the later (csv) contains all tables combined in a machine and humand readable CSV. Then, run the replication script as follows:

```
python /src/evaluation/c_rq2_table.py
```
Running this script will take longer than the other ones, and will print out some warnings. We are aware of them, and they do not influence the results.

After the script finished running, verify that the previously deleted files have been sucessfuly re-created. 

### :heavy_check_mark: Latency Evaluation (Table 7)
Lastly, we will replicate the latencies presented in table 7. Delete the file `generated/results/table_times.tex` and run the script as follows:

```
python /src/evaluation/d_times_table.py
```

Again, the latex table should be re-generated.

:tada::tada: You have successfully replicated all the results presented in the paper :tada::tada:



## Supplement: How to re-generate models and predictions
Our paper discusses the trade-off between local and remote models, with their corresponding supervisors (e.g. uncertainty quantifiers),
and our approach is in-principle agnostic to the actual models and supervisors used. 
Our experiments as described in aboves replication steps are based on a given collection of predictions, uncertainties and time-measurements for existing models mostly taken from the literature, or collected from third-party services such as the OpenAI API. All these can be found in the folders `generated/trained_models/` (custom small models for imdb and issues case studies), `generated/predictions_and_uncertainties` (CSV's of predictions, uncertainties and latentcy measurements for all case studies) and `generated/gpt3` (json-collection of all requests made to OpenAI, allowing to re-run our scripts without having to pay for requests).

While we do not recommend re-running the scripts we used to create these artifacts if not strictly needed (see below), we provide them for completeness.
The scripts may also come in handy for studies extending our experiments. 

- Imdb local predictions: ```python src/prediction_collection/imdb_local.py```
- Imdb remote predictions: ```python src/prediction_collection/imdb_remote.py```
- SQuADv2 local predictions: ```python src/prediction_collection/squadv2_local.py```
- SQuADv2 remote predictions: ```python src/prediction_collection/squadv2_remote.py```
- Issues local predictions: ```python src/prediction_collection/issues_local.py```
- Issues remote predictions: ```python src/prediction_collection/issues_remote.py```
- Imagenet local and remote predictions:  ```python src/prediction_collection/imagenet_local_and_remote.py```

Note the following / Manual Steps: 
- These scripts are not part of our proposed approach, and thus not part of the "replication steps".
- The scripts take quite a long time to run.
- Wherever possible, these scripts will automatically download datasets and models.
- Parts of the scripts have random influences (e.g. model training, latency) and may thus lead to different final results.
- The imagenet case study relies on the large imagenet dataset (>100GB) which has to be manually downloaded and mounted to the docker container for copyright reasons ([read more about this here](https://www.tensorflow.org/datasets/catalog/imagenet2012)).
- Scripts relying on the OpenAI require in principle an OpenAI key, to be pasted into `generated/gpt3/access_token.txt`. However, as we cached all the requests we made as part of our experiments (`generated/gpt3/text-curie-001.json` and `generated/gpt3/text-davinci-003.json`), the scripts `src/prediction_collection/imdb_remote.py` and `src/prediction_collection/squadv2_remote.py` can run without the api key, reading the OpenAI API responses from these files.
- The issues remote model (CatISS) has to be downloaded from [here](https://drive.google.com/drive/folders/1jgV4U41-2acctpc6jH5DWL3fF5V6bKF8) (google drive folder belonging to the [replication package](https://github.com/MalihehIzadi/catiss) of the paper releasing CatISS) and placed in `generated/trained_models/catiss.bin`.



## License
This project is MIT licensed. See the [LICENSE](LICENSE) file for more details.

## Paper
This the registered report for this paper is in-principle accepted at TOSEM. 
For a pre-print, please contact michael.weiss@usi.ch. By the time you are reading this, you may also find the paper on arXiv: "Adopting Two Supervisors for Efficient Use of Large-Scale Remote Deep Neural Networks" by M. Weiss and P. Tonella.

