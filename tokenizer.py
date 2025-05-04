from transformers import AutoTokenizer

sentence = "Hello World"

# load the pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# extract the token ids
token_ids = tokenizer(sentence).input_ids
for id in token_ids:
    a = tokenizer.decode(id)
    # print(a)

colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence: str, tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_ids = tokenizer(sentence).input_ids
    
    print(f"Vocab length: {len(tokenizer)}")
    # Print a colored list of tokens
    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )

text = """
I don't understand what you say right now
"""

# show_tokens(text, "bert-base-cased")
# show_tokens(text, "bert-base-uncased")
show_tokens(text, "Xenova/gpt-4")