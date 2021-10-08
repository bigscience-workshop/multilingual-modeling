import csv
import logging
import datasets
logger = logging.getLogger('ROOT')


_CITATION = """@article{DBLP:journals/corr/abs-2004-14353,
  author    = {Weijia Xu and
               Batool Haider and
               Saab Mansour},
  title     = {End-to-End Slot Alignment and Recognition for Cross-Lingual {NLU}},
  journal   = {CoRR},
  volume    = {abs/2004.14353},
  year      = {2020},
  url       = {https://arxiv.org/abs/2004.14353},
  archivePrefix = {arXiv},
  eprint    = {2004.14353},
  timestamp = {Sun, 03 May 2020 17:39:04 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2004-14353.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}"""

_DESCRIPTION = "Multilingual Atis Datasets."
_LANGS = ["de", "en", "es", "fr", "hi", "ja", "pt", "tr", "zh", "custom_mix"]
_HOMEPAGE = "https://github.com/amazon-research/multiatis"


class MultiAtisConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(MultiAtisConfig, self).__init__(**kwargs)


class MultiAtis(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = []
    BUILDER_CONFIGS = [
        MultiAtisConfig(name=lang, description=f"MultAtis examples in language {lang}")
        for lang in _LANGS
    ]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("int32"),
                "chunks": datasets.Sequence(datasets.Value("string")),
                "chunk_labels": datasets.Sequence(datasets.Value("string")),
                "lang": datasets.Value("string"),
                "class": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, __dl_manager):
        # select train data files
        lang = self.config.name
        if lang == "custom_mix":
            train_data_files = self.config.data_files["train"]
        else:
            train_data_files = [
                _file
                for _file in self.config.data_files["train"]
                if _file.split("_")[-1].split(".tsv")[0] == lang.upper()
            ]
        # select validation data files
        if lang == "custom_mix":
            validation_data_files = self.config.data_files["validation"]
        else:
            validation_data_files = [
                _file
                for _file in self.config.data_files["validation"]
                if _file.split("_")[-1].split(".tsv")[0] == lang.upper()
            ]
        # select test data files
        if lang == "custom_mix":
            test_data_files = self.config.data_files["test"]
        else:
            test_data_files = [
                _file
                for _file in self.config.data_files["test"]
                if _file.split("_")[-1].split(".tsv")[0] == lang.upper()
            ]
        # read data
        train_split = datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={"filepaths": train_data_files},
        )
        validation_split = datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            gen_kwargs={"filepaths": validation_data_files},
        )
        test_split = datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={"filepaths": test_data_files},
        )
        return [train_split, validation_split, test_split]

    def get_clean_intent(self, intent):
        return intent.replace(" ", "_").split("#")[0]

    def _generate_examples(self, filepaths):
        sample_cnt = 0
        err_cnt = 0
        for filepath in filepaths:
            lang = filepath.split("_")[-1].split(".tsv")[0]
            with open(filepath, "r") as tsv_file_ptr:
                tsv_rows = csv.reader(tsv_file_ptr, delimiter="\t")
                for idx, row in enumerate(tsv_rows):
                    if idx == 0:
                        continue
                    assert len(row) == 4
                    _id, utterance, slot_labels, intent = row[0], row[1], row[2], row[3]
                    intent = self.get_clean_intent(intent)
                    try:
                        assert len(utterance.split(' ')) == len(slot_labels.split(' '))
                        yield sample_cnt, {
                            "id": _id,
                            "chunks": utterance.split(' '),
                            "chunk_labels": slot_labels.split(' '),
                            "lang": lang,
                            "class": intent,
                        }
                        sample_cnt += 1
                    except:
                        err_cnt += 1
                        logger.warning("Skipping samples for not matching `len(utterance.split(' ')) == len(slot_labels.split(' '))`, # of sample: {}".format(err_cnt))


###############
# To Load dataset
#
# dataset = datasets.load_dataset(
#         dataset_path_or_name,
#         dataset_config_name,
#         data_files=data_files,
#         cache_dir=data_args.cache_dir,
#     )
#
# dataset_path_or_name = address of this (data_loader/multiatis.py) file
# dataset_config_name = language
# data_files = path_to_multiatis_dataset_folder/MultiATIS++/data/train_dev_test
################