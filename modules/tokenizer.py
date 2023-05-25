import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

def tokenize(dataset , vocab_size:int, save_path:str):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(special_tokens=["[SOS]", "[EOS]", "[UNK]", "[MASK]","[PAD]"], vocab_size=vocab_size)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(dataset.text, trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $A [EOS]",
        special_tokens=[("[SOS]", 0), ("[EOS]", 1)],
    )
    save_path = os.path.join(save_path)
    tokenizer.save(str(save_path))
    print(tokenizer.get_vocab())
    return tokenizer