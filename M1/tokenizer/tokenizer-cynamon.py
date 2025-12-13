from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from corpora import get_corpus_file

TOKENIZER_OUTPUT_FILE = "tokenizers/cynamonka.json"

# 1. Initialize the Tokenizer (BPE model)
tokenizer = Tokenizer(BPE(unk_token="[UNK]")) 

# 2. Set the pre-tokenizer (e.g., split on spaces)
tokenizer.pre_tokenizer = Whitespace()

# 3. Set the Trainer
trainer = BpeTrainer(
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    vocab_size=32000,
    min_frequency=2
)

FILES = [str(f) for f in get_corpus_file("WOLNELEKTURY", "sklepy-cynamonowe.txt")]
print(FILES)

# 4. Train the Tokenizer
tokenizer.train(FILES, trainer=trainer)

# 5. Save the vocabulary and tokenization rules
tokenizer.save(TOKENIZER_OUTPUT_FILE)

for txt in [
    "Litwo! Ojczyzno moja! ty jesteś jak zdrowie.",
    "Jakże mi wesoło!",
    "Jeśli wolisz mieć pełną kontrolę nad tym, które listy są łączone (a to jest bezpieczniejsze, gdy słownik może zawierać inne klucze), po prostu prześlij listę list do spłaszczenia.",
    "W okresie najkrótszych, sennych dni zimowych, ujętych z obu stron, od poranku i od wieczora, w futrzane krawędzie zmierzchów, gdy miasto rozgałęziało się coraz głębiej w labirynty zimowych nocy, z trudem przywoływane przez krótki świt do opamiętania, do powrotu — ojciec mój był już zatracony, zaprzedany, zaprzysiężony tamtej sferze.",
]:
    encoded = tokenizer.encode(txt)
    print("Zakodowany tekst:", encoded.tokens)
    print("ID tokenów:", encoded.ids)
