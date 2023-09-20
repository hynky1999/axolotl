import logging
from axolotl.prompt_tokenizers import PromptTokenizingStrategy
from axolotl.prompters import IGNORE_TOKEN_ID

class ConversationPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for Conversations
    """
    def __init__(
        self,
        prompter,
        tokenizer,
        train_on_inputs: bool = False,
        sequence_len: int = 2048,
    ):
        super().__init__(prompter, tokenizer, train_on_inputs, sequence_len)

    def tokenize_prompt(self, prompt):
        conversation = self.parse_conversation(prompt)
        conversation = self.preprocess_conversation(conversation)
        return self.preprocess(conversation)

    def parse_conversation(self, prompt):
        conversation = prompt["conversations"]
        return conversation

    def preprocess_conversation(self, conversation):
        conv = self.prompter("vicuna")
        roles = {"human": conv.roles[0], "gpt": conv.roles[1], "system": ""}

        # Apply prompt templates
        if roles[conversation[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            # TODO load system prompt if exists
            conversation = conversation[1:]

        for j, sentence in enumerate(conversation):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"role mismatch: {role} vs. {conv.roles[j % 2]}"
            conv.append_message(role, sentence["value"])
        return conv


    def preprocess(self, conv):
        full_prompt = conv.get_prompt()

        tokenized = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            max_length=self.sequence_len,
            truncation=True,
        )
        if self.train_on_inputs:
            return {
                "input_ids": tokenized.input_ids[0].tolist(),
                "attention_mask": tokenized.attention_mask[0].tolist(),
                "labels": tokenized.input_ids[0].tolist(),
            }
        targets = tokenized.input_ids[0].clone()

        # Mask targets. Only compute loss on the assistant outputs.
        # todo allow for difference seps
        sep = conv.sep + conv.roles[1] + ": "
        total_len = len(targets)

        turns = full_prompt.split(conv.sep2)
        cur_len = 1
        targets[:cur_len] = IGNORE_TOKEN_ID
        for turn in turns:
            if turn == "":
                break
            turn_len = len(self.tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            # 1 token for :<space> and 1 token for <bos> at start
            instruction_len = len(self.tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            targets[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        targets[cur_len:] = IGNORE_TOKEN_ID

        if cur_len != total_len:
            targets[:] = IGNORE_TOKEN_ID
            logging.warning(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" (ignored)"
            )

        return {
            "input_ids": tokenized.input_ids[0].tolist(),
            "attention_mask": tokenized.attention_mask[0].tolist(),
            "labels": targets.tolist(),
        }