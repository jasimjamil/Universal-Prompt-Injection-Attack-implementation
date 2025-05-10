import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalPromptInjectionLLM:
    def __init__(
        self,
        model_name: str = "distilgpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
    ):
        """
        Initialize the Universal Prompt Injection Attack class.
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (str): Device to run the model on ('cuda' or 'cpu')
            max_length (int): Maximum sequence length for the model
        """
        self.device = device
        self.max_length = max_length
        
        logger.info(f"Loading model {model_name} on {device}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.attack_tokens = None
        self.original_outputs = {}
        self.attacked_outputs = {}

    def generate_universal_trigger(
        self,
        target_texts: List[str],
        num_trigger_tokens: int = 10,
        num_iterations: int = 100,
        learning_rate: float = 0.1,
    ) -> Tuple[List[str], Dict]:
        """
        Generate universal adversarial triggers using gradient-based optimization.
        
        Args:
            target_texts (List[str]): List of target texts to optimize the trigger for
            num_trigger_tokens (int): Number of tokens in the trigger
            num_iterations (int): Number of optimization iterations
            learning_rate (float): Learning rate for optimization
            
        Returns:
            Tuple[List[str], Dict]: Optimized trigger tokens and training statistics
        """
        logger.info("Generating universal trigger...")
        
        # Initialize random trigger tokens
        vocab_size = self.tokenizer.vocab_size
        trigger_tokens = torch.randint(0, vocab_size, (num_trigger_tokens,)).float()  # Convert to float tensor
        trigger_tokens = trigger_tokens.to(self.device)
        trigger_tokens.requires_grad = True
        
        optimizer = torch.optim.Adam([trigger_tokens], lr=learning_rate)
        stats = {"loss_history": []}
        
        for iteration in tqdm(range(num_iterations)):
            total_loss = 0
            
            for target in target_texts:
                # Tokenize target
                target_ids = self.tokenizer.encode(target, return_tensors="pt").to(self.device)
                
                # Convert trigger tokens to integers for concatenation
                trigger_ids = trigger_tokens.long()
                
                # Concatenate trigger with target
                input_ids = torch.cat([trigger_ids.unsqueeze(0), target_ids], dim=1)
                
                # Forward pass
                outputs = self.model(input_ids)
                logits = outputs.logits
                
                # Calculate loss (maximize probability of target sequence)
                loss = -torch.nn.functional.log_softmax(logits[:, :-1], dim=-1).gather(
                    2, target_ids[:, 1:].unsqueeze(-1)
                ).mean()
                
                total_loss += loss
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            stats["loss_history"].append(total_loss.item())
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}, Loss: {total_loss.item():.4f}")
        
        # Convert optimized tokens to text
        self.attack_tokens = self.tokenizer.convert_ids_to_tokens(trigger_tokens.detach().cpu().numpy())
        
        return self.attack_tokens, stats

    def evaluate_attack(
        self,
        test_texts: List[str],
        target_phrase: str,
    ) -> Dict:
        """
        Evaluate the effectiveness of the universal trigger on test texts.
        
        Args:
            test_texts (List[str]): List of texts to test the trigger on
            target_phrase (str): Target phrase to measure success against
            
        Returns:
            Dict: Evaluation metrics
        """
        if self.attack_tokens is None:
            raise ValueError("No attack tokens found. Run generate_universal_trigger first.")
            
        logger.info("Evaluating attack effectiveness...")
        
        metrics = {
            "success_rate": 0,
            "original_outputs": [],
            "attacked_outputs": []
        }
        
        for text in tqdm(test_texts):
            # Generate output without trigger
            original_output = self.generate_text(text)
            self.original_outputs[text] = original_output
            metrics["original_outputs"].append(original_output)
            
            # Generate output with trigger
            trigger_text = " ".join(self.attack_tokens) + " " + text
            attacked_output = self.generate_text(trigger_text)
            self.attacked_outputs[text] = attacked_output
            metrics["attacked_outputs"].append(attacked_output)
            
            # Check if attack was successful
            if target_phrase.lower() in attacked_output.lower():
                metrics["success_rate"] += 1
                
        metrics["success_rate"] /= len(test_texts)
        logger.info(f"Attack success rate: {metrics['success_rate']:.2%}")
        
        return metrics

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate text from a prompt using the model.
        
        Args:
            prompt (str): Input prompt
            max_new_tokens (int): Maximum number of new tokens to generate
            temperature (float): Sampling temperature
            
        Returns:
            str: Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_attack_tokens(self, filepath: str):
        """Save the attack tokens to a file."""
        if self.attack_tokens is None:
            raise ValueError("No attack tokens found. Run generate_universal_trigger first.")
            
        with open(filepath, "w") as f:
            f.write(" ".join(self.attack_tokens))
            
    def load_attack_tokens(self, filepath: str):
        """Load attack tokens from a file."""
        with open(filepath, "r") as f:
            self.attack_tokens = f.read().strip().split() 