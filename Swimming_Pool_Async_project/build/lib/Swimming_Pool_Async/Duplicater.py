# Duplicater.py
import asyncio
import copy
import torch
from transformers import AutoTokenizer
from enum import Enum, auto
from Swimming_Pool_Async.Process_Controller import Process_Controller

class ProcessStage(Enum):
    DIALOG_GENERATE = auto()
    REWARD_THERAPY_QUALITY = auto()

async def async_print(*args, **kwargs):
    await asyncio.to_thread(print, *args, **kwargs)

class Duplicater:
    def __init__(self, tokenizer_path="/data/jcxy/llm_model/Qwen2-7B-Instruct-AWQ", sensitive_words='', iteration=0):
        # Load tokenizer
        self.tokenizer_path = tokenizer_path
        self.current_score = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # Add special tokens for specific models
        if any(x in self.tokenizer_path for x in ["PsyLLM", "Qwen", "xiaoyun"]):
            self.tokenizer.add_special_tokens({"bos_token": "<|im_start|>"})
            self.tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
            self.tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})
            
        self.sensitive_words = sensitive_words
        if isinstance(self.sensitive_words, str):
            self.sensitive_words = self.sensitive_words.split()
        self.iteration = iteration
        self.llm = ""
        self.llm2 = ""
        self.llm_dict = {}
        self.threshold = 0

    def detect_duplicates(self, text):
        # Encode text using tokenizer
        encoded_text = self.tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        unique_elements, counts = torch.unique(encoded_text, return_counts=True)
        duplicates = unique_elements[counts > 1]
        duplicate_counts = counts[counts > 1]
        return list(duplicates), list(duplicate_counts)
    
    def detect_sensitive_word(self, content):
        if any(word in content for word in self.sensitive_words):
            for word in self.sensitive_words:
                if word in content:
                    print(f"敏感词词汇: {word}")
            return False
        return True
    
    def forward(self, content):
        if isinstance(content, list):
            concatenated_content = "".join(item["content"] for item in content)
            content = concatenated_content
        duplicates, counts = self.detect_duplicates(content)
        if counts and max(counts) > 1600:
            return False
        else:
            return True

    async def process_result(self, result, stage):
        """
        Process individual results, check for duplicates, sensitive words, and length.
        """
        duplicate_mistake = 0
        sensitive_word_mistake = 0
        length_mistake = 0
        cut_length = 1

        if result == "skip" or isinstance(result, str):
            return None
            
        # Extract value based on result structure
        val = ""
        if "chosen" in result:
            val = result.get("chosen", "")
        elif "output" in result:
            val = result.get("output", "") if result["output"] else result.get("instruction", "")
        
        # If val is a list (e.g., conversation history), join it
        if isinstance(val, list):
            val = "".join(str(item.get('content', '')) for item in val if isinstance(item, dict))

        # Uniqueness check
        try:
            is_unique = self.forward(val)
        except Exception as e:
            print("Error during uniqueness check:", val, e)
            is_unique = False

        # Sensitive word check
        has_sensitive = self.detect_sensitive_word(val)

        # Length check
        is_length_ok = len(val) >= cut_length

        # Return result if all checks pass
        if is_unique and has_sensitive and is_length_ok:
            if isinstance(result, list):
                return result[0]
            else:
                return result

        # Error logging
        if not is_unique:
            duplicate_mistake += 1
        elif not has_sensitive:
            sensitive_word_mistake += 1
        elif not is_length_ok:
            length_mistake += 1

        if duplicate_mistake != 0 or sensitive_word_mistake != 0 or length_mistake != 0:
            print(f"\n\n\n{stage} 遇到重复输出 {duplicate_mistake} 条, 敏感词输出 {sensitive_word_mistake} 条, 长度不足输出 {length_mistake} 条 fix....\n\n")
            print(val)
            return None

    async def process_and_filter_async(self, processer: Process_Controller, inputs, stage, max_retries=3, cut_length=1, rollout=4, use_autocot=False):
        """
        Async processing and filtering of input data for DIALOG_GENERATE and REWARD_THERAPY_QUALITY.
        """
        fix_cache = copy.deepcopy(inputs)
        
        # Define processing functions for allowed stages
        process_stages = {
            ProcessStage.DIALOG_GENERATE: processer.process_stage_dialog_generate,
            ProcessStage.REWARD_THERAPY_QUALITY: processer.process_stage_reward_therapy_quality,
        }

        process_stage_func = process_stages.get(stage)
        
        if not process_stage_func:
            print(f"Stage {stage} not supported in this simplified version.")
            yield None
            return

        # Task dispatching logic
        tasks = []
        if stage == ProcessStage.REWARD_THERAPY_QUALITY:
            # REWARD_THERAPY_QUALITY expects unpacked arguments
            tasks = [process_stage_func(item, update_prompt, use_intervent) for (item, update_prompt, use_intervent) in fix_cache]
        else:
            # DIALOG_GENERATE (and default behavior) expects single item
            tasks = [process_stage_func(item) for item in fix_cache]

        # Process tasks as they complete
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            
            if isinstance(result, bool):
                yield result
                continue
            
            if isinstance(result, str):
                if result == "<|_error_|>":
                    yield None
                continue

            # Process the dictionary result
            processed_result = await self.process_result(result, stage)
            yield processed_result