<<<<<<< HEAD
# Universal Prompt Injection Attack on Language Models

This project implements a universal prompt injection attack against large language models, specifically targeting models like DistilGPT-2. The implementation is based on academic research on adversarial attacks against language models.

## Features

- Implementation of Universal Prompt Injection Attack
- Support for various pre-trained language models (tested with DistilGPT-2)
- Evaluation metrics for attack success rate
- Easy-to-use API for generating and testing adversarial triggers
- Modular and reusable code structure

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main implementation is in the `UniversalPromptInjectionLLM` class in `promptinjectionLLM.py`. Here's a basic example of how to use it:

```python
from promptinjectionLLM import UniversalPromptInjectionLLM

# Initialize the attack model
attack_model = UniversalPromptInjectionLLM(
    model_name="distilgpt2",
    max_length=512
)

# Generate universal trigger
attack_tokens, stats = attack_model.generate_universal_trigger(
    target_texts=["your", "target", "texts"],
    num_trigger_tokens=10,
    num_iterations=100
)

# Evaluate the attack
metrics = attack_model.evaluate_attack(
    test_texts=["your", "test", "texts"],
    target_phrase="desired output phrase"
)
```

For a complete example, see `example_attack.py`.

## Implementation Details

The `UniversalPromptInjectionLLM` class implements the following key functionalities:

1. **Model Initialization**: Loads a pre-trained language model and tokenizer
2. **Universal Trigger Generation**: Implements gradient-based optimization to find adversarial triggers
3. **Attack Evaluation**: Measures the success rate of the generated triggers
4. **Text Generation**: Handles text generation with and without triggers

## Key Parameters

- `model_name`: Name of the pre-trained model to use (default: "distilgpt2")
- `num_trigger_tokens`: Number of tokens in the universal trigger
- `num_iterations`: Number of optimization iterations
- `learning_rate`: Learning rate for trigger optimization

## Results

The attack effectiveness can be measured through:
- Success rate of injecting target phrases
- Comparison of original vs. attacked outputs
- Loss history during trigger optimization

## Limitations

- The attack may not work equally well on all language models
- Success rates can vary depending on the target phrase and context
- Computational requirements increase with model size and number of iterations

## References

This implementation is based on research in adversarial attacks on language models. Key references include:
- [List relevant papers and resources]

## License

[Your chosen license] 
=======
# Universal-Prompt-Injection-Attack-implementation
>>>>>>> 011ac8099b1d1af66bfe96312426dee81f63a536
