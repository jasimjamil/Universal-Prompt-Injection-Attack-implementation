from promptinjectionLLM import UniversalPromptInjectionLLM
import logging
from datasets import load_dataset

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize the attack model
    logger.info("Initializing model...")
    attack_model = UniversalPromptInjectionLLM(
        model_name="distilgpt2",  # Using DistilGPT-2 as specified in the requirements
        max_length=512
    )

    # Load example dataset (using a small subset of wikitext for demonstration)
    logger.info("Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Select a few examples for demonstration
    target_texts = dataset["text"][:5]  # Using first 5 examples
    test_texts = dataset["text"][5:10]  # Using next 5 examples for testing
    
    # Filter out empty texts
    target_texts = [text for text in target_texts if text.strip()]
    test_texts = [text for text in test_texts if text.strip()]

    # Define target phrase for the attack
    target_phrase = "This text has been hacked"

    # Generate universal trigger
    logger.info("Generating universal trigger...")
    attack_tokens, stats = attack_model.generate_universal_trigger(
        target_texts=target_texts,
        num_trigger_tokens=10,
        num_iterations=50,  # Reduced for demonstration
        learning_rate=0.1
    )

    # Save the attack tokens
    attack_model.save_attack_tokens("attack_tokens.txt")
    
    # Evaluate the attack
    logger.info("Evaluating attack...")
    metrics = attack_model.evaluate_attack(
        test_texts=test_texts,
        target_phrase=target_phrase
    )

    # Print results
    logger.info("\nAttack Results:")
    logger.info(f"Success Rate: {metrics['success_rate']:.2%}")
    
    # Print example outputs
    logger.info("\nExample Outputs:")
    for i, (orig, attacked) in enumerate(zip(
        metrics['original_outputs'][:2],
        metrics['attacked_outputs'][:2]
    )):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Original: {orig[:100]}...")
        logger.info(f"Attacked: {attacked[:100]}...")

if __name__ == "__main__":
    main() 