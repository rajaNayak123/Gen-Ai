import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("Vocab Size of GPT", encoder.n_vocab)

text = "I am raja nayak"

tokens = encoder.encode(text)
print("The tokens", tokens)

decodedText = encoder.decode(tokens)
print(decodedText)