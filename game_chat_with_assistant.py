# Use a pipeline as a high-level helper
from colorama import Fore
import torch
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import pipeline


class chat_bot_communication:
    llm_pipeline = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16,
                            device_map="auto")

    def prompt(self, prompt):
        messages = [
            {"role": "user", "content": prompt},
        ]

        prompt = self.llm_pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.llm_pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        print(outputs[0]["generated_text"])


if __name__ == "__main__":
    chat = chat_bot_communication()
    while True:
        question = input("Ask me: ")
        chat.prompt(question)

        # Quit conversation
        quit_conversation = input(Fore.LIGHTYELLOW_EX + "\nDo you want to quit out conversation? [y/n]")
        if quit_conversation.lower() == "y":
            print(Fore.BLUE + "\nHope you enjoy our conversation!")
            break
        elif quit_conversation.lower() == "n":
            continue
        else:
            print(Fore.LIGHTGREEN_EX + "\nI did not understand! Therefore I leave this chat. Buy and see you!")
            break


