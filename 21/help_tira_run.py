import copy
import dataclasses
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterable

from overrides import overrides
from transformers import PreTrainedTokenizer, AutoTokenizer, ElectraTokenizer

from allennlp.common.util import sanitize_wordpiece
from allennlp.data.tokenizers.token_class import Token
from allennlp.data.tokenizers.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class PretrainedTransformerTokenizer1(Tokenizer):
    """
    A `PretrainedTransformerTokenizer` uses a model from HuggingFace's
    `transformers` library to tokenize some input text.  This often means wordpieces
    (where `'AllenNLP is awesome'` might get split into `['Allen', '##NL', '##P', 'is',
    'awesome']`), but it could also use byte-pair encoding, or some other tokenization, depending
    on the pretrained model that you're using.
    We take a model name as an input parameter, which we will pass to
    `AutoTokenizer.from_pretrained`.
    We also add special tokens relative to the pretrained model and truncate the sequences.
    This tokenizer also indexes tokens and adds the indexes to the `Token` fields so that
    they can be picked up by `PretrainedTransformerIndexer`.
    Registered as a `Tokenizer` with name "pretrained_transformer".
    # Parameters
    model_name : `str`
        The name of the pretrained wordpiece tokenizer to use.
    add_special_tokens : `bool`, optional, (default=`True`)
        If set to `True`, the sequences will be encoded with the special tokens relative
        to their model.
    max_length : `int`, optional (default=`None`)
        If set to a number, will limit the total sequence returned so that it has a maximum length.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        if tokenizer_kwargs is None:
            tokenizer_kwargs = {}
        else:
            tokenizer_kwargs = tokenizer_kwargs.copy()
        tokenizer_kwargs.setdefault("use_fast", True)
        # Note: Just because we request a fast tokenizer doesn't mean we get one.

        from allennlp.common import cached_transformers

        self.tokenizer = ElectraTokenizer.from_pretrained("/notebook/touche2021/electra_tokenizer")

        self._add_special_tokens = add_special_tokens
        self._max_length = max_length

        self._tokenizer_lowercases = self.tokenizer_lowercases(self.tokenizer)

        try:
            self._reverse_engineer_special_tokens("a", "b", model_name, tokenizer_kwargs)
        except AssertionError:
            # For most transformer models, "a" and "b" work just fine as dummy tokens.  For a few,
            # they don't, and so we use "1" and "2" instead.
            self._reverse_engineer_special_tokens("1", "2", model_name, tokenizer_kwargs)

    def _reverse_engineer_special_tokens(
        self,
        token_a: str,
        token_b: str,
        model_name: str,
        tokenizer_kwargs: Optional[Dict[str, Any]],
    ):
        # storing the special tokens
        self.sequence_pair_start_tokens = []
        self.sequence_pair_mid_tokens = []
        self.sequence_pair_end_tokens = []
        # storing token type ids for the sequences
        self.sequence_pair_first_token_type_id = None
        self.sequence_pair_second_token_type_id = None

        # storing the special tokens
        self.single_sequence_start_tokens = []
        self.single_sequence_end_tokens = []
        # storing token type id for the sequence
        self.single_sequence_token_type_id = None

        # Reverse-engineer the tokenizer for two sequences
        from allennlp.common import cached_transformers

        tokenizer_with_special_tokens = ElectraTokenizer.from_pretrained("/notebook/touche2021/electra_tokenizer")
        dummy_output = tokenizer_with_special_tokens.encode_plus(
            token_a,
            token_b,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )
        if len(dummy_output["token_type_ids"]) != len(dummy_output["input_ids"]):
            logger.warning(
                "Tokenizer library did not return valid token type ids. We will assume they are all zero."
            )
            dummy_output["token_type_ids"] = [0] * len(dummy_output["input_ids"])

        dummy_a = self.tokenizer.encode(token_a, add_special_tokens=False)[0]
        assert dummy_a in dummy_output["input_ids"]
        dummy_b = self.tokenizer.encode(token_b, add_special_tokens=False)[0]
        assert dummy_b in dummy_output["input_ids"]
        assert dummy_a != dummy_b

        seen_dummy_a = False
        seen_dummy_b = False
        for token_id, token_type_id in zip(
            dummy_output["input_ids"], dummy_output["token_type_ids"]
        ):
            if token_id == dummy_a:
                if seen_dummy_a or seen_dummy_b:  # seeing a twice or b before a
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_a = True
                assert (
                    self.sequence_pair_first_token_type_id is None
                    or self.sequence_pair_first_token_type_id == token_type_id
                ), "multiple different token type ids found for the first sequence"
                self.sequence_pair_first_token_type_id = token_type_id
                continue

            if token_id == dummy_b:
                if seen_dummy_b:  # seeing b twice
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_b = True
                assert (
                    self.sequence_pair_second_token_type_id is None
                    or self.sequence_pair_second_token_type_id == token_type_id
                ), "multiple different token type ids found for the second sequence"
                self.sequence_pair_second_token_type_id = token_type_id
                continue

            token = Token(
                tokenizer_with_special_tokens.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.sequence_pair_start_tokens.append(token)
            elif not seen_dummy_b:
                self.sequence_pair_mid_tokens.append(token)
            else:
                self.sequence_pair_end_tokens.append(token)

        assert (
            len(self.sequence_pair_start_tokens)
            + len(self.sequence_pair_mid_tokens)
            + len(self.sequence_pair_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=True)

        # Reverse-engineer the tokenizer for one sequence
        dummy_output = tokenizer_with_special_tokens.encode_plus(
            token_a,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=False,
        )
        if len(dummy_output["token_type_ids"]) != len(dummy_output["input_ids"]):
            logger.warning(
                "Tokenizer library did not return valid token type ids. We will assume they are all zero."
            )
            dummy_output["token_type_ids"] = [0] * len(dummy_output["input_ids"])

        seen_dummy_a = False
        for token_id, token_type_id in zip(
            dummy_output["input_ids"], dummy_output["token_type_ids"]
        ):
            if token_id == dummy_a:
                if seen_dummy_a:
                    raise ValueError("Cannot auto-determine the number of special tokens added.")
                seen_dummy_a = True
                assert (
                    self.single_sequence_token_type_id is None
                    or self.single_sequence_token_type_id == token_type_id
                ), "multiple different token type ids found for the sequence"
                self.single_sequence_token_type_id = token_type_id
                continue

            token = Token(
                tokenizer_with_special_tokens.convert_ids_to_tokens(token_id),
                text_id=token_id,
                type_id=token_type_id,
            )
            if not seen_dummy_a:
                self.single_sequence_start_tokens.append(token)
            else:
                self.single_sequence_end_tokens.append(token)

        assert (
            len(self.single_sequence_start_tokens) + len(self.single_sequence_end_tokens)
        ) == self.tokenizer.num_special_tokens_to_add(pair=False)

    @staticmethod
    def tokenizer_lowercases(tokenizer: PreTrainedTokenizer) -> bool:
        # Huggingface tokenizers have different ways of remembering whether they lowercase or not. Detecting it
        # this way seems like the least brittle way to do it.
        tokenized = tokenizer.tokenize(
            "A"
        )  # Use a single character that won't be cut into word pieces.
        detokenized = " ".join(tokenized)
        return "a" in detokenized

    @overrides
    def tokenize(self, text: str) -> List[Token]:
        """
        This method only handles a single sentence (or sequence) of text.
        """
        max_length = self._max_length
        if max_length is not None and not self._add_special_tokens:
            max_length += self.num_special_tokens_for_sequence()

        encoded_tokens = self.tokenizer.encode_plus(
            text=text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True if max_length is not None else False,
            return_tensors=None,
            return_offsets_mapping=self.tokenizer.is_fast,
            return_attention_mask=False,
            return_token_type_ids=True,
            return_special_tokens_mask=True,
        )
        # token_ids contains a final list with ids for both regular and special tokens
        token_ids, token_type_ids, special_tokens_mask, token_offsets = (
            encoded_tokens["input_ids"],
            encoded_tokens["token_type_ids"],
            encoded_tokens["special_tokens_mask"],
            encoded_tokens.get("offset_mapping"),
        )

        # If we don't have token offsets, try to calculate them ourselves.
        if token_offsets is None:
            token_offsets = self._estimate_character_indices(text, token_ids)

        tokens = []
        for token_id, token_type_id, special_token_mask, offsets in zip(
            token_ids, token_type_ids, special_tokens_mask, token_offsets
        ):
            # In `special_tokens_mask`, 1s indicate special tokens and 0s indicate regular tokens.
            # NOTE: in transformers v3.4.0 (and probably older versions) the docstring
            # for `encode_plus` was incorrect as it had the 0s and 1s reversed.
            # https://github.com/huggingface/transformers/pull/7949 fixed this.
            if not self._add_special_tokens and special_token_mask == 1:
                continue

            if offsets is None or offsets[0] >= offsets[1]:
                start = None
                end = None
            else:
                start, end = offsets

            tokens.append(
                Token(
                    text=self.tokenizer.convert_ids_to_tokens(token_id, skip_special_tokens=False),
                    text_id=token_id,
                    type_id=token_type_id,
                    idx=start,
                    idx_end=end,
                )
            )

        return tokens

    def _estimate_character_indices(
        self, text: str, token_ids: List[int]
    ) -> List[Optional[Tuple[int, int]]]:
        """
        The huggingface tokenizers produce tokens that may or may not be slices from the
        original text.  Differences arise from lowercasing, Unicode normalization, and other
        kinds of normalization, as well as special characters that are included to denote
        various situations, such as "##" in BERT for word pieces from the middle of a word, or
        "Ġ" in RoBERTa for the beginning of words not at the start of a sentence.
        This code attempts to calculate character offsets while being tolerant to these
        differences. It scans through the text and the tokens in parallel, trying to match up
        positions in both. If it gets out of sync, it backs off to not adding any token
        indices, and attempts to catch back up afterwards. This procedure is approximate.
        Don't rely on precise results, especially in non-English languages that are far more
        affected by Unicode normalization.
        """

        token_texts = [
            sanitize_wordpiece(t) for t in self.tokenizer.convert_ids_to_tokens(token_ids)
        ]
        token_offsets: List[Optional[Tuple[int, int]]] = [None] * len(token_ids)
        if self._tokenizer_lowercases:
            text = text.lower()
            token_texts = [t.lower() for t in token_texts]

        min_allowed_skipped_whitespace = 3
        allowed_skipped_whitespace = min_allowed_skipped_whitespace

        text_index = 0
        token_index = 0
        while text_index < len(text) and token_index < len(token_ids):
            token_text = token_texts[token_index]
            token_start_index = text.find(token_text, text_index)

            # Did we not find it at all?
            if token_start_index < 0:
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue

            # Did we jump too far?
            non_whitespace_chars_skipped = sum(
                1 for c in text[text_index:token_start_index] if not c.isspace()
            )
            if non_whitespace_chars_skipped > allowed_skipped_whitespace:
                # Too many skipped characters. Something is wrong. Ignore this token.
                token_index += 1
                # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
                allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
                continue
            allowed_skipped_whitespace = min_allowed_skipped_whitespace

            token_offsets[token_index] = (
                token_start_index,
                token_start_index + len(token_text),
            )
            text_index = token_start_index + len(token_text)
            token_index += 1
        return token_offsets

    def _intra_word_tokenize(
        self, string_tokens: List[str]
    ) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
        tokens: List[Token] = []
        offsets: List[Optional[Tuple[int, int]]] = []
        for token_string in string_tokens:
            wordpieces = self.tokenizer.encode_plus(
                token_string,
                add_special_tokens=False,
                return_tensors=None,
                return_offsets_mapping=False,
                return_attention_mask=False,
            )
            wp_ids = wordpieces["input_ids"]

            if len(wp_ids) > 0:
                offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
                tokens.extend(
                    Token(text=wp_text, text_id=wp_id)
                    for wp_id, wp_text in zip(wp_ids, self.tokenizer.convert_ids_to_tokens(wp_ids))
                )
            else:
                offsets.append(None)
        return tokens, offsets

    @staticmethod
    def _increment_offsets(
        offsets: Iterable[Optional[Tuple[int, int]]], increment: int
    ) -> List[Optional[Tuple[int, int]]]:
        return [
            None if offset is None else (offset[0] + increment, offset[1] + increment)
            for offset in offsets
        ]

    def intra_word_tokenize(
        self, string_tokens: List[str]
    ) -> Tuple[List[Token], List[Optional[Tuple[int, int]]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that tokens[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.
        This function inserts special tokens.
        """
        tokens, offsets = self._intra_word_tokenize(string_tokens)
        tokens = self.add_special_tokens(tokens)
        offsets = self._increment_offsets(offsets, len(self.single_sequence_start_tokens))
        return tokens, offsets

    def intra_word_tokenize_sentence_pair(
        self, string_tokens_a: List[str], string_tokens_b: List[str]
    ) -> Tuple[List[Token], List[Optional[Tuple[int, int]]], List[Optional[Tuple[int, int]]]]:
        """
        Tokenizes each word into wordpieces separately and returns the wordpiece IDs.
        Also calculates offsets such that wordpieces[offsets[i][0]:offsets[i][1] + 1]
        corresponds to the original i-th token.
        This function inserts special tokens.
        """
        tokens_a, offsets_a = self._intra_word_tokenize(string_tokens_a)
        tokens_b, offsets_b = self._intra_word_tokenize(string_tokens_b)
        offsets_b = self._increment_offsets(
            offsets_b,
            (
                len(self.sequence_pair_start_tokens)
                + len(tokens_a)
                + len(self.sequence_pair_mid_tokens)
            ),
        )
        tokens_a = self.add_special_tokens(tokens_a, tokens_b)
        offsets_a = self._increment_offsets(offsets_a, len(self.sequence_pair_start_tokens))

        return tokens_a, offsets_a, offsets_b

    def add_special_tokens(
        self, tokens1: List[Token], tokens2: Optional[List[Token]] = None
    ) -> List[Token]:
        def with_new_type_id(tokens: List[Token], type_id: int) -> List[Token]:
            return [dataclasses.replace(t, type_id=type_id) for t in tokens]

        # Make sure we don't change the input parameters
        tokens2 = copy.deepcopy(tokens2)

        if tokens2 is None:
            return (
                self.single_sequence_start_tokens
                + with_new_type_id(tokens1, self.single_sequence_token_type_id)  # type: ignore
                + self.single_sequence_end_tokens
            )
        else:
            return (
                self.sequence_pair_start_tokens
                + with_new_type_id(tokens1, self.sequence_pair_first_token_type_id)  # type: ignore
                + self.sequence_pair_mid_tokens
                + with_new_type_id(tokens2, self.sequence_pair_second_token_type_id)  # type: ignore
                + self.sequence_pair_end_tokens
            )

    def num_special_tokens_for_sequence(self) -> int:
        return len(self.single_sequence_start_tokens) + len(self.single_sequence_end_tokens)

    def num_special_tokens_for_pair(self) -> int:
        return (
            len(self.sequence_pair_start_tokens)
            + len(self.sequence_pair_mid_tokens)
            + len(self.sequence_pair_end_tokens)
        )
    
    
import logging
import math
from typing import Optional, Tuple, Dict, Any

from overrides import overrides

import torch
import torch.nn.functional as F
from transformers import XLNetConfig

#from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from allennlp.nn.util import batched_index_select

logger = logging.getLogger(__name__)


#@TokenEmbedder.register("pretrained_transformer")
class PretrainedTransformerEmbedder1(TokenEmbedder):

    authorized_missing_keys = [r"position_ids$"]

    def __init__(
        self,
        model_name: str,
        *,
        max_length: int = None,
        sub_module: str = None,
        train_parameters: bool = True,
        eval_mode: bool = False,
        last_layer_only: bool = True,
        override_weights_file: Optional[str] = None,
        override_weights_strip_prefix: Optional[str] = None,
        load_weights: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        from allennlp.common import cached_transformers

        self.transformer_model = cached_transformers.get(
            model_name,
            True,
            override_weights_file=override_weights_file,
            override_weights_strip_prefix=override_weights_strip_prefix,
            load_weights=load_weights,
            **(transformer_kwargs or {}),
        )

        if gradient_checkpointing is not None:
            self.transformer_model.config.update({"gradient_checkpointing": gradient_checkpointing})

        self.config = self.transformer_model.config
        if sub_module:
            assert hasattr(self.transformer_model, sub_module)
            self.transformer_model = getattr(self.transformer_model, sub_module)
        self._max_length = max_length

        # I'm not sure if this works for all models; open an issue on github if you find a case
        # where it doesn't work.
        self.output_dim = self.config.hidden_size

        self._scalar_mix: Optional[ScalarMix] = None
        if not last_layer_only:
            self._scalar_mix = ScalarMix(self.config.num_hidden_layers)
            self.config.output_hidden_states = True

        tokenizer = PretrainedTransformerTokenizer1(
            model_name,
            tokenizer_kwargs=tokenizer_kwargs,
        )

        try:
            if self.transformer_model.get_input_embeddings().num_embeddings != len(
                tokenizer.tokenizer
            ):
                self.transformer_model.resize_token_embeddings(len(tokenizer.tokenizer))
        except NotImplementedError:
            # Can't resize for transformers models that don't implement base_model.get_input_embeddings()
            logger.warning(
                "Could not resize the token embedding matrix of the transformer model. "
                "This model does not support resizing."
            )

        self._num_added_start_tokens = len(tokenizer.single_sequence_start_tokens)
        self._num_added_end_tokens = len(tokenizer.single_sequence_end_tokens)
        self._num_added_tokens = self._num_added_start_tokens + self._num_added_end_tokens

        self.train_parameters = train_parameters
        if not train_parameters:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        self.eval_mode = eval_mode
        if eval_mode:
            self.transformer_model.eval()

    @overrides
    def train(self, mode: bool = True):
        self.training = mode
        for name, module in self.named_children():
            if self.eval_mode and name == "transformer_model":
                module.eval()
            else:
                module.train(mode)
        return self

    @overrides
    def get_output_dim(self):
        return self.output_dim

    def _number_of_token_type_embeddings(self):
        if isinstance(self.config, XLNetConfig):
            return 3  # XLNet has 3 type ids
        elif hasattr(self.config, "type_vocab_size"):
            return self.config.type_vocab_size
        else:
            return 0

    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters
        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, e.g. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: `[batch_size, num_wordpieces if max_length is None else num_segment_concat_wordpieces]`.
        segment_concat_mask: `Optional[torch.BoolTensor]`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
        # Returns
        `torch.Tensor`
            Shape: `[batch_size, num_wordpieces, embedding_size]`.
        """
        # Some of the huggingface transformers don't support type ids at all and crash when you supply
        # them. For others, you can supply a tensor of zeros, and if you don't, they act as if you did.
        # There is no practical difference to the caller, so here we pretend that one case is the same
        # as another case.
        if type_ids is not None:
            max_type_id = type_ids.max()
            if max_type_id == 0:
                type_ids = None
            else:
                if max_type_id >= self._number_of_token_type_embeddings():
                    raise ValueError("Found type ids too large for the chosen transformer model.")
                assert token_ids.shape == type_ids.shape

        fold_long_sequences = self._max_length is not None and token_ids.size(1) > self._max_length
        if fold_long_sequences:
            batch_size, num_segment_concat_wordpieces = token_ids.size()
            token_ids, segment_concat_mask, type_ids = self._fold_long_sequences(
                token_ids, segment_concat_mask, type_ids
            )

        transformer_mask = segment_concat_mask if self._max_length is not None else mask
        assert transformer_mask is not None
        # Shape: [batch_size, num_wordpieces, embedding_size],
        # or if self._max_length is not None:
        # [batch_size * num_segments, self._max_length, embedding_size]

        # We call this with kwargs because some of the huggingface models don't have the
        # token_type_ids parameter and fail even when it's given as None.
        # Also, as of transformers v2.5.1, they are taking FloatTensor masks.
        parameters = {"input_ids": token_ids, "attention_mask": transformer_mask.float()}
        if type_ids is not None:
            parameters["token_type_ids"] = type_ids

        transformer_output = self.transformer_model(**parameters)
        if self._scalar_mix is not None:
            # The hidden states will also include the embedding layer, which we don't
            # include in the scalar mix. Hence the `[1:]` slicing.
            hidden_states = transformer_output.hidden_states[1:]
            embeddings = self._scalar_mix(hidden_states)
        else:
            embeddings = transformer_output.last_hidden_state

        if fold_long_sequences:
            embeddings = self._unfold_long_sequences(
                embeddings, segment_concat_mask, batch_size, num_segment_concat_wordpieces
            )

        return embeddings

    def _fold_long_sequences(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, torch.LongTensor, Optional[torch.LongTensor]]:
        """
        We fold 1D sequences (for each element in batch), returned by `PretrainedTransformerIndexer`
        that are in reality multiple segments concatenated together, to 2D tensors, e.g.
        [ [CLS] A B C [SEP] [CLS] D E [SEP] ]
        -> [ [ [CLS] A B C [SEP] ], [ [CLS] D E [SEP] [PAD] ] ]
        The [PAD] positions can be found in the returned `mask`.
        # Parameters
        token_ids: `torch.LongTensor`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
            num_segment_concat_wordpieces is num_wordpieces plus special tokens inserted in the
            middle, i.e. the length of: "[CLS] A B C [SEP] [CLS] D E F [SEP]" (see indexer logic).
        mask: `torch.BoolTensor`
            Shape: `[batch_size, num_segment_concat_wordpieces]`.
            The mask for the concatenated segments of wordpieces. The same as `segment_concat_mask`
            in `forward()`.
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_segment_concat_wordpieces].
        # Returns:
        token_ids: `torch.LongTensor`
            Shape: [batch_size * num_segments, self._max_length].
        mask: `torch.BoolTensor`
            Shape: [batch_size * num_segments, self._max_length].
        """
        num_segment_concat_wordpieces = token_ids.size(1)
        num_segments = math.ceil(num_segment_concat_wordpieces / self._max_length)  # type: ignore
        padded_length = num_segments * self._max_length  # type: ignore
        length_to_pad = padded_length - num_segment_concat_wordpieces

        def fold(tensor):  # Shape: [batch_size, num_segment_concat_wordpieces]
            # Shape: [batch_size, num_segments * self._max_length]
            tensor = F.pad(tensor, [0, length_to_pad], value=0)
            # Shape: [batch_size * num_segments, self._max_length]
            return tensor.reshape(-1, self._max_length)

        return fold(token_ids), fold(mask), fold(type_ids) if type_ids is not None else None

    def _unfold_long_sequences(
        self,
        embeddings: torch.FloatTensor,
        mask: torch.BoolTensor,
        batch_size: int,
        num_segment_concat_wordpieces: int,
    ) -> torch.FloatTensor:
        """
        We take 2D segments of a long sequence and flatten them out to get the whole sequence
        representation while remove unnecessary special tokens.
        [ [ [CLS]_emb A_emb B_emb C_emb [SEP]_emb ], [ [CLS]_emb D_emb E_emb [SEP]_emb [PAD]_emb ] ]
        -> [ [CLS]_emb A_emb B_emb C_emb D_emb E_emb [SEP]_emb ]
        We truncate the start and end tokens for all segments, recombine the segments,
        and manually add back the start and end tokens.
        # Parameters
        embeddings: `torch.FloatTensor`
            Shape: [batch_size * num_segments, self._max_length, embedding_size].
        mask: `torch.BoolTensor`
            Shape: [batch_size * num_segments, self._max_length].
            The mask for the concatenated segments of wordpieces. The same as `segment_concat_mask`
            in `forward()`.
        batch_size: `int`
        num_segment_concat_wordpieces: `int`
            The length of the original "[ [CLS] A B C [SEP] [CLS] D E F [SEP] ]", i.e.
            the original `token_ids.size(1)`.
        # Returns:
        embeddings: `torch.FloatTensor`
            Shape: [batch_size, self._num_wordpieces, embedding_size].
        """

        def lengths_to_mask(lengths, max_len, device):
            return torch.arange(max_len, device=device).expand(
                lengths.size(0), max_len
            ) < lengths.unsqueeze(1)

        device = embeddings.device
        num_segments = int(embeddings.size(0) / batch_size)
        embedding_size = embeddings.size(2)

        # We want to remove all segment-level special tokens but maintain sequence-level ones
        num_wordpieces = num_segment_concat_wordpieces - (num_segments - 1) * self._num_added_tokens

        embeddings = embeddings.reshape(
            batch_size, num_segments * self._max_length, embedding_size  # type: ignore
        )
        mask = mask.reshape(batch_size, num_segments * self._max_length)  # type: ignore
        # We assume that all 1s in the mask precede all 0s, and add an assert for that.
        # Open an issue on GitHub if this breaks for you.
        # Shape: (batch_size,)
        seq_lengths = mask.sum(-1)
        if not (lengths_to_mask(seq_lengths, mask.size(1), device) == mask).all():
            raise ValueError(
                "Long sequence splitting only supports masks with all 1s preceding all 0s."
            )
        # Shape: (batch_size, self._num_added_end_tokens); this is a broadcast op
        end_token_indices = (
            seq_lengths.unsqueeze(-1) - torch.arange(self._num_added_end_tokens, device=device) - 1
        )

        # Shape: (batch_size, self._num_added_start_tokens, embedding_size)
        start_token_embeddings = embeddings[:, : self._num_added_start_tokens, :]
        # Shape: (batch_size, self._num_added_end_tokens, embedding_size)
        end_token_embeddings = batched_index_select(embeddings, end_token_indices)

        embeddings = embeddings.reshape(batch_size, num_segments, self._max_length, embedding_size)
        embeddings = embeddings[
            :, :, self._num_added_start_tokens : embeddings.size(2) - self._num_added_end_tokens, :
        ]  # truncate segment-level start/end tokens
        embeddings = embeddings.reshape(batch_size, -1, embedding_size)  # flatten

        # Now try to put end token embeddings back which is a little tricky.

        # The number of segment each sequence spans, excluding padding. Mimicking ceiling operation.
        # Shape: (batch_size,)
        num_effective_segments = (seq_lengths + self._max_length - 1) // self._max_length
        # The number of indices that end tokens should shift back.
        num_removed_non_end_tokens = (
            num_effective_segments * self._num_added_tokens - self._num_added_end_tokens
        )
        # Shape: (batch_size, self._num_added_end_tokens)
        end_token_indices -= num_removed_non_end_tokens.unsqueeze(-1)
        assert (end_token_indices >= self._num_added_start_tokens).all()
        # Add space for end embeddings
        embeddings = torch.cat([embeddings, torch.zeros_like(end_token_embeddings)], 1)
        # Add end token embeddings back
        embeddings.scatter_(
            1, end_token_indices.unsqueeze(-1).expand_as(end_token_embeddings), end_token_embeddings
        )

        # Now put back start tokens. We can do this before putting back end tokens, but then
        # we need to change `num_removed_non_end_tokens` a little.
        embeddings = torch.cat([start_token_embeddings, embeddings], 1)

        # Truncate to original length
        embeddings = embeddings[:, :num_wordpieces, :]
        return embeddings
    
class PretrainedTransformerMismatchedEmbedder1(TokenEmbedder):
    """
    Use this embedder to embed wordpieces given by `PretrainedTransformerMismatchedIndexer`
    and to pool the resulting vectors to get word-level representations.
    Registered as a `TokenEmbedder` with name "pretrained_transformer_mismatched".
    # Parameters
    model_name : `str`
        The name of the `transformers` model to use. Should be the same as the corresponding
        `PretrainedTransformerMismatchedIndexer`.
    max_length : `int`, optional (default = `None`)
        If positive, folds input token IDs into multiple segments of this length, pass them
        through the transformer model independently, and concatenate the final representations.
        Should be set to the same value as the `max_length` option on the
        `PretrainedTransformerMismatchedIndexer`.
    train_parameters: `bool`, optional (default = `True`)
        If this is `True`, the transformer weights get updated during training.
    last_layer_only: `bool`, optional (default = `True`)
        When `True` (the default), only the final layer of the pretrained transformer is taken
        for the embeddings. But if set to `False`, a scalar mix of all of the layers
        is used.
    gradient_checkpointing: `bool`, optional (default = `None`)
        Enable or disable gradient checkpointing.
    tokenizer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/tokenization_utils.py#L691)
        for `AutoTokenizer.from_pretrained`.
    transformer_kwargs: `Dict[str, Any]`, optional (default = `None`)
        Dictionary with
        [additional arguments](https://github.com/huggingface/transformers/blob/155c782a2ccd103cf63ad48a2becd7c76a7d2115/transformers/modeling_utils.py#L253)
        for `AutoModel.from_pretrained`.
    """  # noqa: E501

    def __init__(
        self,
        model_name: str,
        max_length: int = None,
        train_parameters: bool = True,
        last_layer_only: bool = True,
        gradient_checkpointing: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        transformer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        # The matched version v.s. mismatched
        self._matched_embedder = PretrainedTransformerEmbedder1(
            model_name,
            max_length=max_length,
            train_parameters=train_parameters,
            #override_weights_file = '/home/katana21/runs/embedder.ph',
            override_weights_file = 'embedder.ph',
            last_layer_only=last_layer_only,
            gradient_checkpointing=gradient_checkpointing,
            tokenizer_kwargs=tokenizer_kwargs,
            transformer_kwargs=transformer_kwargs,
        )

    @overrides
    def get_output_dim(self):
        return self._matched_embedder.get_output_dim()


    @overrides
    def forward(
        self,
        token_ids: torch.LongTensor,
        mask: torch.BoolTensor,
        offsets: torch.LongTensor,
        wordpiece_mask: torch.BoolTensor,
        type_ids: Optional[torch.LongTensor] = None,
        segment_concat_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:  # type: ignore
        """
        # Parameters
        token_ids: `torch.LongTensor`
            Shape: [batch_size, num_wordpieces] (for exception see `PretrainedTransformerEmbedder`).
        mask: `torch.BoolTensor`
            Shape: [batch_size, num_orig_tokens].
        offsets: `torch.LongTensor`
            Shape: [batch_size, num_orig_tokens, 2].
            Maps indices for the original tokens, i.e. those given as input to the indexer,
            to a span in token_ids. `token_ids[i][offsets[i][j][0]:offsets[i][j][1] + 1]`
            corresponds to the original j-th token from the i-th batch.
        wordpiece_mask: `torch.BoolTensor`
            Shape: [batch_size, num_wordpieces].
        type_ids: `Optional[torch.LongTensor]`
            Shape: [batch_size, num_wordpieces].
        segment_concat_mask: `Optional[torch.BoolTensor]`
            See `PretrainedTransformerEmbedder`.
        # Returns
        `torch.Tensor`
            Shape: [batch_size, num_orig_tokens, embedding_size].
        """
        # Shape: [batch_size, num_wordpieces, embedding_size].
        embeddings = self._matched_embedder(
            token_ids, wordpiece_mask, type_ids=type_ids, segment_concat_mask=segment_concat_mask
        )

        # span_embeddings: (batch_size, num_orig_tokens, max_span_length, embedding_size)
        # span_mask: (batch_size, num_orig_tokens, max_span_length)
        span_embeddings, span_mask = util.batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings

        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)

        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

        return orig_embeddings