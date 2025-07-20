#NLP chatbot voice activated

![Alt Text](image.png)


## 🔍 Chatbot logic

```python
else:
```

* Enters the `else` block if previous conditions were not met.

```python
    if ai.text == "ERROR":
```

* Checks if the input text (likely transcribed speech) is `"ERROR"`.

```python
        res = "Sorry, come again?"
```

* Sets a fallback response when speech recognition fails.

```python
    else:
```

* Proceeds to AI generation if the input is valid.

```python
        input_ids = tokenizer.encode(ai.text + tokenizer.eos_token, return_tensors='pt')
```

* Encodes the input text along with the end-of-sequence token into PyTorch tensors.

```python
        with torch.no_grad():
```

* Disables gradient calculations for inference to optimize performance.

```python
            output_ids = model.generate(
```

* Begins generating output tokens from the model.

```python
                input_ids,
                max_length=1000,
```

* Uses the encoded input and limits output to 1000 tokens.

```python
                pad_token_id=tokenizer.eos_token_id,
```

* Uses the EOS token for padding.

```python
                no_repeat_ngram_size=3,
```

* Prevents repeating 3-word phrases in the generated response.

```python
                do_sample=True,
```

* Enables sampling for more varied, creative responses.

```python
                top_k=100,
```

* Considers only the top 100 probable tokens during sampling.

```python
                top_p=0.7,
```

* Applies nucleus sampling, selecting tokens with cumulative probability ≤ 0.7.

```python
                temperature=0.8
```

* Controls randomness; lower values make output more predictable.

```python
            )
```

* Ends the `generate()` method call.

```python
        res = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
```

* Decodes only the generated part (excluding input) and removes special tokens.

```python
        if not res.strip():
```

* Checks if the response is empty or just whitespace.

```python
            res = "I'm not sure how to respond to that."
```

* Provides a fallback reply if the generated output is empty.

---

