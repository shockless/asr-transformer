import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import BPEDecoder

def tokenize(dataset , vocab_size:int, save_path:str):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[MASK]", "[SOS]", "[EOS]", "[UNK]", "[PAD]"], vocab_size=vocab_size, end_of_word_suffix="[EOF]")
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(dataset.text, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[("[SOS]", 1), ("[EOS]", 2)],
    )
    tokenizer.decoder = BPEDecoder(suffix="[EOF]")
    save_path = os.path.join(save_path)
    tokenizer.save(str(save_path))
    print(tokenizer.get_vocab())
    return tokenizer