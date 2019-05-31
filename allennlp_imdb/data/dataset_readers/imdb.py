from typing import Dict
import logging

import os.path as osp
from pathlib import Path
import tarfile
from itertools import chain

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)


@DatasetReader.register('imdb')
class ImdbDatasetReader(DatasetReader):

    TAR_URL = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    TRAIN_DIR = 'aclImdb/train'
    TEST_DIR = 'aclImdb/test'

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)

        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        tar_path = cached_path(self.TAR_URL)
        tf = tarfile.open(tar_path, 'r')
        cache_dir = Path(osp.dirname(tar_path))
        if not (cache_dir / self.TRAIN_DIR).exists() and not (cache_dir / self.TEST_DIR).exists():
            tf.extractall(cache_dir)

        if file_path == 'train':
            pos_dir = osp.join(self.TRAIN_DIR, 'pos')
            neg_dir = osp.join(self.TRAIN_DIR, 'neg')
        elif file_path == 'test':
            pos_dir = osp.join(self.TEST_DIR, 'pos')
            neg_dir = osp.join(self.TEST_DIR, 'neg')
        else:
            raise ValueError(f"only 'train' and 'test' are valid for 'file_path', but '{file_path}' is given.")
        path = chain(Path(cache_dir.joinpath(pos_dir)).glob('*.txt'),
                     Path(cache_dir.joinpath(neg_dir)).glob('*.txt'))

        for p in path:
            yield self.text_to_instance(p.read_text(), 0 if 'pos' in str(p) else 1)

    @overrides
    def text_to_instance(self, string: str, label: int) -> Instance:
        fields: Dict[str, Field] = {}
        tokens = self._tokenizer.tokenize(string)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        fields['label'] = LabelField(label, skip_indexing=True)
        return Instance(fields)
