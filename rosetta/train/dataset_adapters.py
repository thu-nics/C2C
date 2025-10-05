"""
Simple dataset adapter for converting InstructCoder to chat format
"""

from typing import List, Dict, Any, Optional, Union, Callable
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import inspect
import os
import hashlib
# Dataset Registry System
DATASET_REGISTRY = {}

def register_dataset(cls=None, name=None):
    """
    Register a dataset class in the global registry.
    Can be used as a decorator with or without arguments.
    
    Args:
        cls: The class to register
        name: Optional name to register the class under. If None, uses the class name.
        
    Returns:
        The registered class
    """
    def _register(cls):
        dataset_name = name if name is not None else cls.__name__
        DATASET_REGISTRY[dataset_name] = cls
        # Also register with lowercase name for case-insensitive lookup
        DATASET_REGISTRY[dataset_name.lower()] = cls
        return cls
    
    # Called as @register_dataset
    if cls is not None:
        return _register(cls)
    
    # Called as @register_dataset() or @register_dataset(name="DatasetName")
    return _register


def capture_init_args(cls):
    """
    Decorator to capture initialization arguments of a dataset class.
    
    Args:
        cls: The class to decorate
        
    Returns:
        The decorated class with automatic init args capture
    """
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        # Store all initialization arguments
        self._init_args = {}
        
        # Get parameter names from the original __init__ method
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        # Map positional args to parameter names
        for i, arg in enumerate(args):
            if i < len(param_names):
                self._init_args[param_names[i]] = arg
        
        # Add keyword args
        self._init_args.update(kwargs)
        
        # Call the original __init__
        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls


# Unified batch filtering functions


def create_text_length_filter(
    max_length: int,
    text_extractor: Callable[[Dict[str, Any]], str],
    tokenizer: Optional[Any] = None,
    use_tokens: bool = False
):
    """
    Unified text length filter that can handle both word count and token count filtering.
    
    Args:
        max_length: Maximum allowed length (words or tokens)
        text_extractor: Function that extracts text from a single sample
        tokenizer: Tokenizer for token counting (required if use_tokens=True)
        use_tokens: If True, count tokens; if False, count words
        
    Returns:
        Filter function that can be used with dataset.filter(batched=True)
    """
    if use_tokens and tokenizer is None:
        raise ValueError("Tokenizer must be provided when use_tokens=True")
    
    def _text_length_filter_batch(batch):
        batch_size = len(next(iter(batch.values())))
        samples = [{key: values[i] for key, values in batch.items()} for i in range(batch_size)]
        try:
            texts = [text_extractor(sample) for sample in samples]
            if use_tokens:
                if hasattr(tokenizer, 'apply_chat_template') and any(isinstance(t, list) for t in texts):
                    rendered = []
                    for t in texts:
                        if isinstance(t, list):
                            rendered.append(tokenizer.apply_chat_template(t, tokenize=False, add_generation_prompt=False))
                        else:
                            rendered.append(str(t))
                    tokenized = tokenizer(rendered, add_special_tokens=False)
                else:
                    tokenized = tokenizer([str(t) for t in texts], add_special_tokens=False)
                lengths = [len(ids) for ids in tokenized["input_ids"]]
            else:
                lengths = [len(str(t).split()) for t in texts]
            return [length <= max_length for length in lengths]
        except Exception as e:
            print(f"Error in text length filter: {e}")
            return [False] * batch_size
    
    return _text_length_filter_batch


def create_field_value_filter(target_value: Any, field_name: str, comparison: str = 'equal'):
    """
    Unified field value filter for exact matching, language filtering, etc.
    
    Args:
        target_value: Value to compare against
        field_name: Field name to check
        comparison: Type of comparison ('equal', 'not_equal', 'in', 'not_in')
        
    Returns:
        Filter function that can be used with dataset.filter(batched=True)
    """
    def _field_value_filter_batch(batch):
        field_values = batch.get(field_name, [])
        
        if comparison == 'equal':
            return [value == target_value for value in field_values]
        elif comparison == 'not_equal':
            return [value != target_value for value in field_values]
        elif comparison == 'in':
            return [value in target_value for value in field_values]
        elif comparison == 'not_in':
            return [value not in target_value for value in field_values]
        else:
            raise ValueError(f"Unsupported comparison: {comparison}")
    
    return _field_value_filter_batch


def create_modulo_filter(mod_base: int, exclude_values: Union[int, List[int]], field_name: str = '_id'):
    """
    Unified modulo filter for ID-based filtering.
    
    Args:
        mod_base: Modulo base
        exclude_values: Value(s) to exclude (can be single int or list)
        field_name: Field name containing the ID
        
    Returns:
        Filter function that can be used with dataset.filter(batched=True)
    """
    if isinstance(exclude_values, int):
        exclude_values = [exclude_values]
    
    def _modulo_filter_batch(batch):
        ids = batch.get(field_name, [])
        results = []
        
        for _id in ids:
            try:
                # Try numeric conversion first
                id_num = int(_id)
                mod_result = id_num % mod_base
            except (ValueError, TypeError):
                # Use hash for non-numeric IDs
                id_hash = hash(str(_id))
                mod_result = id_hash % mod_base
            
            results.append(mod_result not in exclude_values)
        
        return results
    
    return _modulo_filter_batch


def create_conversation_length_filter(min_messages: int, text_field: str = 'conversations'):
    """
    Unified conversation length filter for OpenHermes-style datasets.
    
    Args:
        min_messages: Minimum number of messages required (excluding system messages)
        text_field: Field name containing the conversation
        
    Returns:
        Filter function that can be used with dataset.filter(batched=True)
    """
    def _conversation_length_filter_batch(batch):
        conversations_list = batch.get(text_field, [])
        results = []
        
        for conversations in conversations_list:
            try:
                # Extract messages (excluding system)
                message_count = 0
                for msg in conversations:
                    role = msg.get('from') or msg.get('role')
                    if role in ('human', 'user', 'gpt', 'assistant'):
                        message_count += 1
                
                results.append(message_count > min_messages)
            except Exception:
                results.append(False)
        
        return results
    
    return _conversation_length_filter_batch


# Text extraction functions for common dataset patterns
def extract_mmlu_text(sample: Dict[str, Any], question_field: str = 'question', choices_field: str = 'choices') -> str:
    """Extract text from MMLU-style samples"""
    question = sample.get(question_field, '')
    choices = sample.get(choices_field, [])
    
    # Handle both list and dict formats for choices
    if isinstance(choices, dict):
        choices_text = choices.get('text', [])
    else:
        choices_text = choices
    
    return (str(question) + " " + " ".join(map(str, choices_text))).strip()


def extract_chat_text(sample: Dict[str, Any], input_field: str = 'input', 
                     context_field: str = 'context', answers_field: str = 'answers') -> List[Dict[str, str]]:
    """Extract chat messages from LongBench-style samples"""
    input_text = str(sample.get(input_field, ''))
    context = str(sample.get(context_field, ''))
    answers = sample.get(answers_field, [])
    
    assistant_message = answers[0] if answers and len(answers) > 0 else "No answer provided"
    
    # Build complete chat format
    if context:
        human_message = f"Context: {context}\n\nInstruction: {input_text}"
    else:
        human_message = f"Instruction: {input_text}"
    
    return [
        {"role": "user", "content": human_message.strip()},
        {"role": "assistant", "content": assistant_message.strip()}
    ]


def extract_conversation_text(sample: Dict[str, Any], text_field: str = 'conversations') -> str:
    """Extract text from OpenHermes-style conversation samples"""
    conversations = sample.get(text_field, [])
    
    if conversations and len(conversations) > 0:
        return conversations[0].get('value', '')
    return ''


def extract_first_user_message(sample: Dict[str, Any], text_field: str = 'conversations') -> str:
    """Extract the first human/user message from conversation-style samples."""
    conversations = sample.get(text_field, [])
    for msg in conversations:
        role = msg.get('from') or msg.get('role')
        if role in ('human', 'user'):
            return str(msg.get('value', ''))
    # Fallback to first message if role tags are missing
    if conversations:
        return str(conversations[0].get('value', ''))
    return ''


def extract_first_assistant_message(sample: Dict[str, Any], text_field: str = 'conversations') -> str:
    """Extract the first gpt/assistant message from conversation-style samples."""
    conversations = sample.get(text_field, [])
    for msg in conversations:
        role = msg.get('from') or msg.get('role')
        if role in ('gpt', 'assistant'):
            return str(msg.get('value', ''))
    # Fallback to second message if present
    if len(conversations) > 1:
        return str(conversations[1].get('value', ''))
    return ''


def extract_openhermes_messages(sample: Dict[str, Any], text_field: str = 'conversations') -> List[Dict[str, str]]:
    """Build chat messages excluding system; include all human/user and gpt/assistant in order."""
    conversation = sample.get(text_field, [])
    messages: List[Dict[str, str]] = []
    for msg in conversation:
        role = msg.get('from') or msg.get('role')
        if role == 'system':
            continue
        if role in ('human', 'user'):
            messages.append({"role": "user", "content": str(msg.get('value', '')).strip()})
        elif role in ('gpt', 'assistant'):
            messages.append({"role": "assistant", "content": str(msg.get('value', ''))})
    return messages


def extract_instruction_text(sample: Dict[str, Any], instruction_field: str = 'instruction', 
                           inputs_field: str = 'inputs') -> str:
    """Extract text from Inkuba-style instruction samples"""
    instruction = sample.get(instruction_field)
    inputs = sample.get(inputs_field, '')
    
    if instruction is not None:
        return str(instruction) + "\n\n" + str(inputs)
    else:
        return str(inputs)


def extract_chat_pair_text(sample: Dict[str, Any], user_field: str = 'inputs', 
                          assistant_field: str = 'targets') -> List[Dict[str, str]]:
    """Extract chat messages from Aya-style samples"""
    user_text = str(sample.get(user_field, ''))
    assistant_text = str(sample.get(assistant_field, ''))
    
    return [
        {"role": "user", "content": user_text.strip()},
        {"role": "assistant", "content": assistant_text.strip()}
    ]



def extract_dolly_chat_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract chat messages from Dolly-style samples.

    Fields:
      - instruction: str
      - context: str (may be empty)
      - response: str
      - category: optional, may be empty/missing
    """
    instruction = str(sample.get('instruction', '')).strip()
    context = str(sample.get('context', '') or '').strip()
    response = str(sample.get('response', '')).strip()

    if context:
        user_message = f"{context}\n\n{instruction}"
    else:
        user_message = f"{instruction}"

    return [
        {"role": "user", "content": user_message.strip()},
        {"role": "assistant", "content": response}
    ]


def extract_mmmlu_chat_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    """Extract chat messages from MMMLU-style samples (OpenAI/MMMLU)."""
    choice_labels = ['A', 'B', 'C', 'D']

    template = (
            "Jibu kwa usahihi swali lifuatalo:\n\n"
            "{{question}}\n\n"
            "Chaguo:\n"
            "{{choices}}\n\n"
            "Maelekezo:\n"
            "- Soma swali na chaguo zote kwa makini.\n"
            "- Chagua jibu sahihi zaidi kati ya yaliyotolewa.\n"
            "- Jibu TU kwa herufi (A, B, C, D) inayolingana na jibu sahihi.\n"
            "- Usijumuishe maelezo, maandishi ya ziada, au alama yoyote ya uakifishaji.\n\n"
            "Jibu lako:"
        )

    choices_text = ""
    for label in choice_labels:
        content = sample.get(label, '')
        choices_text += f"{label}. {content}\n"

    user_prompt = template.replace("{{choices}}", choices_text).replace("{{question}}", str(sample.get('Question', '')))

    correct_label = sample.get('Answer', '')
    correct_content = sample.get(correct_label, '')
    assistant_response = f"**Jibu lako: {correct_label}. {correct_content}.**"

    return [
        {"role": "user", "content": user_prompt.strip()},
        {"role": "assistant", "content": assistant_response}
    ]




def apply_batch_filters(dataset, filters: list, filter_descriptions: list = None, 
                       batch_size: int = 4096, combine_filters: bool = True,
                       num_proc: Optional[int] = None):
    """
    Apply multiple filters using native batched filtering for maximum performance.
    
    Args:
        dataset: Dataset to filter
        filters: List of batched filter functions
        filter_descriptions: Optional list of descriptions for logging
        batch_size: Batch size for filtering operations
        combine_filters: If True, combine all filters into a single batched operation
        
    Returns:
        Filtered dataset and original length
    """
    if not filters:
        return dataset, len(dataset)
    
    original_len = len(dataset)
    
    if combine_filters and len(filters) > 1:
        # Combine all filters into a single batched operation for maximum efficiency
        def _combined_batch_filter(batch):
            # Get results from all filters
            filter_results = []
            for filter_func in filters:
                filter_results.append(filter_func(batch))
            
            # Combine results with AND logic
            combined_results = []
            batch_size = len(filter_results[0]) if filter_results else 0
            
            for i in range(batch_size):
                combined_results.append(all(result[i] for result in filter_results))
            
            return combined_results
        
        # Apply combined filter in a single pass
        filtered_dataset = dataset.filter(
            _combined_batch_filter,
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc if num_proc and (num_proc or 0) > 1 else None,
            desc="Combined batch filtering"
        )
        
        # Print filtering results
        final_len = len(filtered_dataset)
        if original_len != final_len:
            print(f"Applied combined batch filtering: {original_len} -> {final_len} samples")
            if filter_descriptions:
                for desc in filter_descriptions:
                    print(f"  - {desc}")
    
    else:
        # Apply each filter sequentially with batched processing
        current_dataset = dataset
        
        for i, (filter_func, desc) in enumerate(zip(filters, filter_descriptions or [''] * len(filters))):
            pre_filter_len = len(current_dataset)
            
            current_dataset = current_dataset.filter(
                filter_func,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc if num_proc and (num_proc or 0) > 1 else None,
                desc=f"Filtering: {desc}" if desc else f"Filter {i+1}"
            )
            
            post_filter_len = len(current_dataset)
            if desc and pre_filter_len != post_filter_len:
                print(f"  - {desc}: {pre_filter_len} -> {post_filter_len} samples")
        
        filtered_dataset = current_dataset
        final_len = len(filtered_dataset)
        if original_len != final_len:
            print(f"Applied sequential batch filtering: {original_len} -> {final_len} samples")
    
    return filtered_dataset, original_len


def generate_kv_cache_index(instruction_length: int, full_length: int) -> torch.tensor:
    """
    Generate KV cache index for the input sequence.
    
    Args:
        instruction_length: Length of the instruction tokens
        full_length: Total length of the full conversation tokens
        
    Returns:
        Tensor with KV cache index
    """
    assert instruction_length <= full_length

    instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(instruction_length - 1, 1)
    label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(full_length - instruction_length + 1, 1)

    kv_cache_index = torch.cat([instruction_index, label_index], dim=0)  # shape: (seq_len, 2)

    return kv_cache_index


"""
Instruction dataset

Convert any form of inputs to standard message format
"""

@register_dataset
@capture_init_args
class LongBenchChatDataset(Dataset):
    """LongBench数据集转换为LongBench原始格式"""
    
    def __init__(self, split: str = "test", num_samples: Optional[int] = None,
                 dataset_name: Optional[str] = None, language: Optional[str] = None,
                 max_word_count: Optional[int] = None, max_length: Optional[int] = 14000,
                 use_longbench_e: bool = True, filter_mod4: bool = True):
        """
        初始化LongBench数据集
        
        Args:
            split: 数据集分割 ("test" - LongBench主要使用test分割)
            num_samples: 使用的样本数量 (None表示全部)
            dataset_name: 特定数据集名称 (None表示所有数据集)
            language: 语言过滤 ("en" 或 "zh")
            max_word_count: 最大词数限制（用于英文文本）
            max_length: 最大字符长度限制
            use_longbench_e: 是否使用LongBench-E版本
            filter_mod4: 是否过滤_id mod4余1的样本
        """
        print(f"Loading LongBench{' -E' if use_longbench_e else ''} dataset (split: {split}, dataset: {dataset_name})...")
        
        # LongBench包含的数据集列表
        longbench_datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", 
            "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", 
            "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", 
            "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"
        ]
        
        longbench_e_datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", 
            "multi_news", "trec", "triviaqa", "samsum", "passage_count", 
            "passage_retrieval_en", "lcc", "repobench-p"
        ]
        
        target_datasets = longbench_e_datasets if use_longbench_e else longbench_datasets
        
        # 定义LongBench提示模板
        self.dataset_prompt_formats = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}
        
        # 定义不使用聊天模板的任务
        #self.no_chat_template_tasks = ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
        self.no_chat_template_tasks=['']
        self.use_longbench_e = use_longbench_e
        self.max_length = max_length

        if dataset_name:
            if dataset_name not in target_datasets:
                raise ValueError(f"Dataset {dataset_name} not found in LongBench{' -E' if use_longbench_e else ''}")
            target_datasets = [dataset_name]
            self.current_evaluating_subject = dataset_name
        else:
            self.current_evaluating_subject = None
        
        # 加载所有选定的数据集
        all_data = []
        for dataset in target_datasets:
            try:
                dataset_suffix = f"{dataset}_e" if use_longbench_e else dataset
                data = load_dataset('THUDM/LongBench', dataset_suffix, split=split)
                print(f"  Loaded {len(data)} samples from {dataset}")
                
                # 添加数据集名称标识
                data = data.map(lambda x: {"dataset_source": dataset})
                all_data.append(data)
            except Exception as e:
                print(f"Warning: Failed to load {dataset}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No datasets were successfully loaded")
        

        from datasets import concatenate_datasets
        self.dataset = concatenate_datasets(all_data)
        




        # mod4!=1
        if filter_mod4:
            original_len = len(self.dataset)
            
            def _mod4_not_1(example):
                _id = example.get('_id', '')
                id_hash = int(hashlib.sha256(str(_id).encode('utf-8')).hexdigest(), 16)
                
                return id_hash % 4 != 1
            
            self.dataset = self.dataset.filter(_mod4_not_1)
            print(f"Filtered by _id mod4 != 1: {original_len} -> {len(self.dataset)} samples")
        
        # 限制样本数量
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))
            
        print(f"Loaded total {len(self.dataset)} samples from LongBench{' -E' if use_longbench_e else ''}")    
    def __len__(self):
        return len(self.dataset)
    
    def _format_longbench_example(self, example: Dict[str, Any], tokenizer: AutoTokenizer) -> str:

        # 1. 确定任务类型
        dataset_source = example.get('dataset_source', '')
        if self.current_evaluating_subject:
            current_subject = self.current_evaluating_subject
        else:
            current_subject = dataset_source
            
        # 仅当字符串以"_e"结尾时才替换
        import re
        subject = re.sub(r"_e$", "", current_subject) if self.use_longbench_e else current_subject
        
        # 2. 获取提示模板
        if subject not in self.dataset_prompt_formats:
            subject = "narrativeqa"  # 默认模板
        prompt_format = self.dataset_prompt_formats[subject]
        
        # 3. 直接使用**example展开所有字段
        raw_prompt = prompt_format.format(**example)
        
        # 4. 超长截断逻辑
        tokenized_raw = tokenizer(raw_prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_raw) > self.max_length:
            half_len = int(self.max_length / 2)
            raw_prompt = tokenizer.decode(tokenized_raw[:half_len], skip_special_tokens=True) + \
                        tokenizer.decode(tokenized_raw[-half_len:], skip_special_tokens=True)
        
        # 5. 应用Chat Template

        final_prompt = raw_prompt
        print(len(tokenized_raw))
        return final_prompt
    
    def __getitem__(self, idx):

        sample = self.dataset[idx]
        
        # 格式化样本
        formatted_prompt = self._format_longbench_example(sample, self.tokenizer)
        
        # 提取答案
        answers = sample.get('answers', [])
        assistant_message = answers[0] if answers and len(answers) > 0 else "No answer provided"
        
        return [
            {
                "role": "user",
                "content": formatted_prompt.strip()
            },
            {
                "role": "assistant", 
                "content": assistant_message.strip()
            }
        ]

@register_dataset
@capture_init_args
class MMLUChatDataset(Dataset):
    """Simple MMLU dataset converted to chat format"""

    def __init__(self, split: str = "train", num_samples: Optional[int] = None, max_word_count: Optional[int] = None):
        """
        Initialize the dataset

        Args:
            split: Dataset split
            num_samples: Number of samples to use (None for all)
            max_word_count: If set, drop samples whose question + all choices exceed this word count
        """
        print(f"Loading MMLU dataset (split: {split})...")
        # Load dataset
        dataset = load_dataset("cais/mmlu", "all")
        dataset = dataset[split]

        # Ensure we have a proper Dataset object
        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        # Limit samples if specified
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))
            
        # Apply total token length filtering on full chat (user + assistant)
        if max_word_count is not None:
            # Use a small tokenizer for speed; total token length = chat(user+assistant)
            self._mmlu_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            extractor = lambda sample: self._build_chat_messages(sample)
            filters = [create_text_length_filter(max_word_count, extractor, self._mmlu_tokenizer, use_tokens=True)]
            filter_descriptions = [f"Token count filter (full chat): max {max_word_count}"]
            self.dataset, _ = apply_batch_filters(self.dataset, filters, filter_descriptions)

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return self._build_chat_messages(sample)

    def _build_chat_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        choice_labels = ['A', 'B', 'C', 'D']
        question = sample.get('question', '')
        choices_list = sample.get('choices', [])
        user_prompt = f"Question: {question}\n\nChoices:\n"
        for i, choice in enumerate(choices_list):
            label = choice_labels[i] if i < len(choice_labels) else chr(65 + i)
            user_prompt += f"{label}. {choice}\n"
        ans_idx = sample.get('answer', 0)
        if isinstance(ans_idx, str) and ans_idx.isdigit():
            ans_idx = int(ans_idx)
        ans_label = choice_labels[ans_idx] if 0 <= int(ans_idx) < len(choice_labels) else chr(65 + int(ans_idx))
        assistant_text = f"The correct answer is {ans_label}."
        return [
            {"role": "user", "content": user_prompt.strip()},
            {"role": "assistant", "content": assistant_text.strip()},
        ]

@register_dataset
@capture_init_args
class MMLUCotChatDataset(Dataset):
    """Simple MMLUCot dataset converted to chat format"""

    def __init__(self, split: str = "train", num_samples: Optional[int] = None):
        """
        Initialize the dataset

        Args:
            split: Dataset split
            num_samples: Number of samples to use (None for all)
        """
        print(f"Loading MMLUCot dataset (split: {split})...")
        # Load dataset
        dataset = load_dataset("Brench/MMLU-Pro-CoT-Train-43K")
        dataset = dataset[split]

        # Ensure we have a proper Dataset object
        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        # Limit samples if specified
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        user_prompt = sample['question'] + "\n"

        assistant_response = sample['chain_of_thoughts']

        return [
            {
                "role": "user",
                "content": user_prompt.strip()
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]

@register_dataset
@capture_init_args
class LLMGeneratedChatDataset(Dataset):
    """Simple LLM Generated dataset converted to chat format"""

    def __init__(self, split: str = "train", num_samples: Optional[int] = None, data_path: str = "./teacher_datasets/output/dataset_finished", max_word_count: Optional[int] = None):
        """
        Initialize the dataset

        Args:
            split: Dataset split
            num_samples: Number of samples to use (None for all)
        """
        print(f"Loading LLMGeneratedCot dataset (split: {split})...")
        # Load dataset
        dataset = load_from_disk(data_path)

        # Ensure we have a proper Dataset object
        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        
        if max_word_count is not None:
            original_len = len(self.dataset)
            half = max_word_count // 2
            def _under_token_limit(batch):
                q = tokenizer(batch["input_text"], add_special_tokens=False, padding=False, truncation=False)
                a = tokenizer(batch["model_response"], add_special_tokens=False, padding=False, truncation=False)
                return [
                    (len(q_ids) <= half) and (len(q_ids) + len(a_ids) <= max_word_count)
                    for q_ids, a_ids in zip(q["input_ids"], a["input_ids"])
                ]

            self.dataset = self.dataset.filter(
                _under_token_limit,
                batched=True,
                batch_size=2048,                    # 视显存/内存调大
                num_proc=min(8, os.cpu_count() or 1),
                load_from_cache_file=True,
                desc=f"Filter max_word_count={max_word_count}",
            )
            print(f"Filtered by max_word_count={max_word_count}: {original_len} -> {len(self.dataset)} samples")

        # Limit samples if specified
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        input_text = sample.get('input_text', '') or ''

        # Parse question and choices from input_text, which is expected to contain a
        # "Choices:" section followed by labeled options like "A. ..."
        def _parse_question_and_choices(text: str):
            lines = (text or '').splitlines()
            # Find the line index for "Choices:" (case-insensitive, ignoring spaces)
            choices_idx = -1
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('choices'):
                    choices_idx = i
                    break

            if choices_idx == -1:
                # Fallback: no explicit Choices header found
                question_part = text.strip()
                return question_part, ''

            question_part = '\n'.join(lines[:choices_idx]).strip()

            # Collect labeled choices until blank line or instruction-like line
            collected = []
            for raw in lines[choices_idx + 1:]:
                s = raw.strip()
                if not s:
                    # Stop on first blank after having collected at least one choice
                    if collected:
                        break
                    else:
                        continue
                lower = s.lower()
                # Stop when hitting instruction section common in prompts
                if lower.startswith('instructions:') or lower.startswith("let's ") or lower.startswith('you must'):
                    break
                # Accept formats like "A. ..." or "A) ..."
                if len(s) >= 3 and s[0] in 'ABCDEFGHIJ' and s[1] in ').' and s[2] == ' ':
                    collected.append(s)
                else:
                    # If we've started collecting and this line doesn't look like a choice, stop
                    if collected:
                        break
                    # Otherwise ignore preamble noise
                    continue

            choices_block = '\n'.join(collected).strip()
            return question_part, choices_block

        question, choices_block = _parse_question_and_choices(input_text)

        # Rebuild user prompt using the evaluation CoT template
        template = """Accurately answer the following question:

{{question}}

Choices:
{{choices}}

Instructions:
- Carefully read the question and all options.
- Let's think step by step and you must explain your reasoning briefly.
- Then give the final answer.
- Keep your response within 150 words."""

        filled_prompt = (
            template
            .replace("{{question}}", question or '')
            .replace("{{choices}}", choices_block or '')
        )

        user_prompt = filled_prompt.strip() + "\n"

        assistant_response = sample['model_response']

        return [
            {
                "role": "user",
                "content": user_prompt.strip()
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]

@register_dataset
@capture_init_args
class OpenBookChatDataset(Dataset):
    """Simple OpenBook dataset converted to chat format"""

    def __init__(self, split: str = "train", num_samples: Optional[int] = None):
        """
        Initialize the dataset

        Args:
            split: Dataset split
            num_samples: Number of samples to use (None for all)
        """
        print(f"Loading OpenBook dataset (split: {split})...")
        # Load dataset
        dataset = load_dataset("allenai/openbookqa", "main")
        dataset = dataset[split]

        # Ensure we have a proper Dataset object
        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")

        # Limit samples if specified
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        choice_labels = ['A', 'B', 'C', 'D']

        user_prompt = (
            f"Question: {sample['question_stem']}\n\n"
            f"Choices:\n"
        )
        for idx, choice in enumerate(sample['choices']['text']):
            label = choice_labels[idx]
            user_prompt += f"{label}. {choice}\n"

        correct_label = sample["answerKey"]
        assistant_response = f"The correct answer is {correct_label}."

        return [
            {
                "role": "user",
                "content": user_prompt.strip()
            },
            {
                "role": "assistant",
                "content": assistant_response
            }
        ]

@register_dataset
@capture_init_args
class OpenHermesChatDataset(Dataset):
    """Simple general dataset converted to chat format"""

    def __init__(self, split: str = "train", num_samples: Optional[int] = None, max_word_count: Optional[int] = None, min_conversation_turns: int = 0):
        """
        Initialize the dataset

        Args:
            split: Dataset split
            num_samples: Number of samples to use (None for all)
            max_word_count: Maximum token count for filtering
            min_conversation_turns: Minimum number of conversation turns (default 3 for multi-turn conversations)
        """
        print(f"Loading OpenHermes dataset (split: {split})...")
        # Load dataset
        dataset = load_dataset("teknium/OpenHermes-2.5")
        dataset = dataset[split]

        # Ensure we have a proper Dataset object
        if hasattr(dataset, 'select'):
            self.dataset = dataset
        else:
            raise ValueError(f"Unexpected dataset type: {type(dataset)}")
        
        # Limit samples if specified
        if num_samples and num_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(num_samples))

        # Apply filters
        filters = []
        filter_descriptions = []

        # Filter by minimum conversation length (exclude conversations with <= 2 messages)
        if min_conversation_turns > 0:
            filters.append(create_conversation_length_filter(min_conversation_turns - 1, 'conversations'))
            filter_descriptions.append(f"Conversation length filter: min {min_conversation_turns} messages (multi-turn only)")

        # Apply conversation-level token count filtering (all messages combined <= max_word_count)
        if max_word_count is not None:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            extractor = lambda sample: extract_openhermes_messages(sample, 'conversations')
            filters.append(create_text_length_filter(max_word_count, extractor, tokenizer, use_tokens=True))
            filter_descriptions.append(f"Token count filter: max {max_word_count}")

        # Apply all filters
        if filters:
            self.dataset, _ = apply_batch_filters(self.dataset, filters, filter_descriptions, num_proc=8)

        print(f"Loaded {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return extract_openhermes_messages(sample, 'conversations')
    
"""
Chat dataset

Convert standard message format to input_ids and labels
"""
class ChatDataset(Dataset):
    """Dataset for chat format training with HuggingFace Trainer compatibility"""
    
    def __init__(self, chat_dataset, tokenizer: AutoTokenizer, max_length: int = 32768):
        self.chat_dataset = chat_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.chat_dataset)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        messages = self.chat_dataset[idx]
        
        # Get instruction (first message)
        instruction = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Get full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Tokenize instruction and full text
        instruction_tokens = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # Truncate if necessary
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        # Create labels (-100 for instruction tokens, actual tokens for response)
        labels = [-100] * len(instruction_tokens) + full_tokens[len(instruction_tokens):]
        # labels = [-100] * (len(full_tokens) - 4) + full_tokens[-4:]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        
        kv_cache_index = generate_kv_cache_index(len(instruction_tokens), len(full_tokens))
        # kv_cache_index = generate_kv_cache_index(len(full_tokens)-4, len(full_tokens))
        # kv_cache_index = generate_kv_cache_index(len(full_tokens) + 1, len(full_tokens))

        return {
            "input_ids": full_tokens,
            "labels": labels,
            "kv_cache_index": kv_cache_index
        }


class AlignedChatDataset(Dataset):
    """Dataset that precomputes aligned inputs for SLM/LLM using a TokenAligner"""
    
    def __init__(self, instruct_dataset: Dataset, aligner: Any, max_length: int = 32768):
        self.dataset = instruct_dataset
        self.aligner = aligner
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        messages = self.dataset[idx]

        # Build aligned sequences and section map
        details = self.aligner.align_chat_messages(messages, add_generation_prompt=False, return_details=True)
        slm_ids: List[int] = details['slm_ids_padded']
        llm_ids: List[int] = details['llm_ids_padded']
        sections = details['sections']

        slm_pad_mask = torch.tensor(details['slm_padding_mask'])
        llm_pad_mask = torch.tensor(details['llm_padding_mask'])
        message_mask = torch.tensor(details['message_mask'])

        # Determine instruction boundary as start of the last message section
        instr_end = 0
        for sec_idx in range(len(sections) - 1, -1, -1):
            sec = sections[sec_idx]
            if sec['type'] == 'message':
                instr_end = sec['slm_range'][0]
                break

        # Labels: follow ChatDataset policy (-100 for instruction-only, supervise the rest)
        labels = [-100] * instr_end + slm_ids[instr_end:]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]

        # Truncate inputs if needed
        if len(slm_ids) > self.max_length:
            slm_ids = slm_ids[:self.max_length]
            # Truncate padding mask accordingly
            slm_pad_mask = slm_pad_mask[:self.max_length]
        if len(llm_ids) > self.max_length:
            llm_ids = llm_ids[:self.max_length]
            llm_pad_mask = llm_pad_mask[:self.max_length]

        # KV cache index based on instruction length
        kv_cache_index = generate_kv_cache_index(instr_end, len(slm_ids))
        # Addtionally mask non-message parts
        kv_cache_index[~message_mask] = torch.tensor([[-1,0]])

        return {
            "input_ids": [slm_ids, llm_ids],
            "labels": labels,
            "kv_cache_index": kv_cache_index,
            "messages": messages,
            # Per-model aligned inputs (per-sample, pre-batch)
            "model_padding_mask": [slm_pad_mask, llm_pad_mask],
        }


class BaselineChatDataset(Dataset):
    """Simple dataset for baseline model training without Rosetta-specific features"""
    
    def __init__(self, chat_dataset, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.chat_dataset = chat_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.chat_dataset)
    
    def __getitem__(self, idx):
        messages = self.chat_dataset[idx]
        
        # Get instruction (first message)
        instruction = self.tokenizer.apply_chat_template(
            messages[:1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # Get full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Tokenize instruction and full text
        instruction_tokens = self.tokenizer(instruction, add_special_tokens=False)["input_ids"]
        full_tokens = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]
        
        # Truncate if necessary
        if len(full_tokens) > self.max_length:
            full_tokens = full_tokens[:self.max_length]
        
        # Create labels (-100 for instruction tokens, actual tokens for response)
        labels = [-100] * len(instruction_tokens) + full_tokens[len(instruction_tokens):]
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]

        return {
            "input_ids": full_tokens,
            "labels": labels,
        }

"""
Data collator

Batch chat data to model input
"""

class RosettaDataCollator:
    """Improved data collator for RosettaModel training with cleaner logic"""

    def __init__(self, slm_tokenizer: AutoTokenizer, llm_tokenizer: AutoTokenizer = None, 
                 pad_to_multiple_of: Optional[int] = None, max_length: Optional[int] = None, 
                 aligner: Optional[Any] = None, do_alignment: bool = False):
        """
        Initialize the collator.
        
        Args:
            slm_tokenizer: Small language model tokenizer
            llm_tokenizer: Large language model tokenizer (optional)
            pad_to_multiple_of: Pad sequence length to multiple of this value
            max_length: Maximum sequence length
            aligner: Alignment module (if needed)
            do_alignment: Whether to perform alignment
        """
        self.slm_tokenizer = slm_tokenizer
        self.llm_tokenizer = llm_tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.max_length = max_length
        self.aligner = aligner
        self.do_alignment = do_alignment
        
        if self.do_alignment:
            assert self.aligner is not None, "Aligner must be provided if do_alignment is True"
        
        # Store padding token IDs for different models
        self.slm_pad_token_id = self.slm_tokenizer.pad_token_id
        self.llm_pad_token_id = self.llm_tokenizer.pad_token_id if self.llm_tokenizer else self.slm_pad_token_id

    def _normalize_input_format(self, feature: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize input format to handle both single and dual model inputs.
        
        Args:
            feature: Input feature dictionary
            
        Returns:
            Normalized feature with consistent format
        """
        # Normalize input_ids: ensure it's always a list of tensors
        input_ids = feature['input_ids']
        if isinstance(input_ids, list) and len(input_ids) > 0:
            if isinstance(input_ids[0], list):
                # Case: [[ids1], [ids2]] -> convert to list of tensors
                input_ids_tensors = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
            else:
                # Case: [id1, id2, ...] -> single model case
                input_ids_tensors = [torch.tensor(input_ids, dtype=torch.long)]
        else:
            # Fallback: assume single model
            input_ids_tensors = [torch.tensor(input_ids, dtype=torch.long)]
        
        # Normalize attention_mask
        attention_masks = []
        if "model_padding_mask" in feature:
            # Use model-specific padding masks
            for model_padding_mask in feature["model_padding_mask"]:
                attention_masks.append((~model_padding_mask).float())
        else:
            # Generate default attention masks
            for input_tensor in input_ids_tensors:
                attention_masks.append(torch.ones(len(input_tensor), dtype=torch.float))
        
        return {
            'input_ids': input_ids_tensors,
            'attention_mask': attention_masks,
            'labels': torch.tensor(feature['labels'], dtype=torch.long),
            'kv_cache_index': feature['kv_cache_index'],
            'position_ids': torch.arange(len(feature['labels']), dtype=torch.long)
        }

    def _split_into_sections(self, normalized_feature: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split sequence into sections based on kv_cache_index changes.
        
        Args:
            normalized_feature: Normalized feature dictionary
            
        Returns:
            List of sections
        """
        kv_idx = normalized_feature['kv_cache_index']
        
        # Find change points in kv_cache_index
        change_points = [0]
        for i in range(1, kv_idx.size(0)):
            if not torch.equal(kv_idx[i], kv_idx[i - 1]):
                change_points.append(i)
        change_points.append(kv_idx.size(0))
        
        # Create sections
        sections = []
        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i + 1]
            section = {
                'input_ids': [ids[start:end] for ids in normalized_feature['input_ids']],
                'attention_mask': [mask[start:end] for mask in normalized_feature['attention_mask']],
                'labels': normalized_feature['labels'][start:end],
                'kv_cache_index': normalized_feature['kv_cache_index'][start:end],
                'position_ids': normalized_feature['position_ids'][start:end]
            }
            sections.append(section)
        
        return sections

    def _pad_sections(self, all_sections: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Pad sections to ensure uniform structure across batch.
        
        Args:
            all_sections: List of section lists for each sample
            
        Returns:
            Padded batch dictionary
        """
        max_sections = max(len(sections) for sections in all_sections)
        num_models = len(all_sections[0][0]['input_ids']) if all_sections else 1
        
        # Initialize output structure - keep models separate throughout
        padded_output = {
            'input_ids_per_model': [[] for _ in range(num_models)],  # One list per model
            'attention_mask_per_model': [[] for _ in range(num_models)],  # One list per model
            'labels': [],
            'kv_cache_index': [],
            'position_ids': []
        }
        
        # Process each section index
        for sec_idx in range(max_sections):
            section_data = self._collect_section_data(all_sections, sec_idx, num_models)
            padded_section = self._pad_single_section(section_data, num_models)
            
            # Add to output - keep models separate
            for model_idx in range(num_models):
                padded_output['input_ids_per_model'][model_idx].append(
                    padded_section['input_ids_per_model'][model_idx])
                padded_output['attention_mask_per_model'][model_idx].append(
                    padded_section['attention_mask_per_model'][model_idx])
            
            padded_output['labels'].append(padded_section['labels'])
            padded_output['kv_cache_index'].append(padded_section['kv_cache_index'])
            padded_output['position_ids'].append(padded_section['position_ids'])
        
        # Concatenate sections and finalize
        return self._finalize_output(padded_output, num_models, len(all_sections))

    def _collect_section_data(self, all_sections: List[List[Dict[str, Any]]], 
                            sec_idx: int, num_models: int) -> Dict[str, List]:
        """Collect data for a specific section across all samples."""
        # Separate collections for each model to avoid confusion
        section_data = {
            'input_ids_per_model': [[] for _ in range(num_models)],  # [[slm_seqs], [llm_seqs]]
            'attention_mask_per_model': [[] for _ in range(num_models)],
            'labels': [],
            'kv_cache_index': [],
            'position_ids': []
        }
        
        for sample_sections in all_sections:
            # Some samples may have fewer sections; create default empty tensors when missing
            if sec_idx < len(sample_sections):
                sec = sample_sections[sec_idx]
                for model_idx in range(num_models):
                    section_data['input_ids_per_model'][model_idx].append(sec['input_ids'][model_idx])
                    section_data['attention_mask_per_model'][model_idx].append(sec['attention_mask'][model_idx])
                section_data['labels'].append(sec['labels'])
                section_data['kv_cache_index'].append(sec['kv_cache_index'])
                section_data['position_ids'].append(sec['position_ids'])
            else:
                # Default empty tensors; downstream pad_sequence will pad appropriately
                for model_idx in range(num_models):
                    section_data['input_ids_per_model'][model_idx].append(torch.tensor([], dtype=torch.long))
                    section_data['attention_mask_per_model'][model_idx].append(torch.tensor([], dtype=torch.float))
                section_data['labels'].append(torch.tensor([], dtype=torch.long))
                section_data['kv_cache_index'].append(torch.empty((0, 2), dtype=torch.long))
                section_data['position_ids'].append(torch.tensor([], dtype=torch.long))
                
        return section_data

    def _pad_single_section(self, section_data: Dict[str, List], num_models: int) -> Dict[str, Any]:
        """Pad tensors within a single section."""
        # Pad input_ids separately for each model with their respective pad tokens
        padded_input_ids_per_model = []
        padded_attention_mask_per_model = []
        
        for model_idx in range(num_models):
            pad_token_id = self.slm_pad_token_id if model_idx == 0 else self.llm_pad_token_id
            
            # Pad input_ids for this model
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(
                section_data['input_ids_per_model'][model_idx], 
                batch_first=True, 
                padding_value=pad_token_id
            )
            padded_input_ids_per_model.append(padded_input_ids)
            
            # Pad attention_mask for this model
            padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
                section_data['attention_mask_per_model'][model_idx],
                batch_first=True,
                padding_value=0
            )
            padded_attention_mask_per_model.append(padded_attention_mask)
        
        # Standard padding for other tensors
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            section_data['labels'], batch_first=True, padding_value=-100)
        padded_kv_cache = torch.nn.utils.rnn.pad_sequence(
            section_data['kv_cache_index'], batch_first=True, padding_value=-1)
        padded_position_ids = torch.nn.utils.rnn.pad_sequence(
            section_data['position_ids'], batch_first=True, padding_value=0)
        
        return {
            'input_ids_per_model': padded_input_ids_per_model,  # Keep separate per model
            'attention_mask_per_model': padded_attention_mask_per_model,  # Keep separate per model
            'labels': padded_labels,
            'kv_cache_index': padded_kv_cache,
            'position_ids': padded_position_ids,
            'num_models': num_models
        }

    def _finalize_output(self, padded_output: Dict[str, List], 
                        num_models: int, batch_size: int) -> Dict[str, Any]:
        """Finalize the output by concatenating sections - keep models separate throughout."""
        final_output = {}
        
        # Handle input_ids and attention_mask - keep separate per model
        if num_models == 1:
            # Single model case: concatenate sections for the single model
            final_output['input_ids'] = torch.cat(padded_output['input_ids_per_model'][0], dim=1)
            final_output['attention_mask'] = torch.cat(padded_output['attention_mask_per_model'][0], dim=1)
        else:
            # Multi-model case: keep as list of tensors, one per model
            final_output['input_ids'] = [
                torch.cat(padded_output['input_ids_per_model'][model_idx], dim=1) 
                for model_idx in range(num_models)
            ]
            final_output['attention_mask'] = [
                torch.cat(padded_output['attention_mask_per_model'][model_idx], dim=1)
                for model_idx in range(num_models)
            ]
        
        # Concatenate other tensors normally
        final_output['labels'] = torch.cat(padded_output['labels'], dim=1)
        final_output['position_ids'] = torch.cat(padded_output['position_ids'], dim=1)
        final_output['kv_cache_index'] = padded_output['kv_cache_index']  # Keep as list of sections
        
        return final_output

    def _apply_length_constraints(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Apply max_length truncation if specified."""
        if self.max_length is None:
            return output
        
        # Determine current sequence length
        if isinstance(output['input_ids'], list):
            seq_length = output['input_ids'][0].size(1)
        else:
            seq_length = output['input_ids'].size(1)
        
        if seq_length <= self.max_length:
            return output
        
        # Truncate sequences
        if isinstance(output['input_ids'], list):
            output['input_ids'] = [ids[:, :self.max_length] for ids in output['input_ids']]
            output['attention_mask'] = [mask[:, :self.max_length] for mask in output['attention_mask']]
        else:
            output['input_ids'] = output['input_ids'][:, :self.max_length]
            output['attention_mask'] = output['attention_mask'][:, :self.max_length]
        
        output['labels'] = output['labels'][:, :self.max_length]
        output['position_ids'] = output['position_ids'][:, :self.max_length]
        
        # Truncate kv_cache_index sections appropriately
        output['kv_cache_index'] = self._truncate_kv_cache_sections(
            output['kv_cache_index'], self.max_length)
        
        return output

    def _truncate_kv_cache_sections(self, kv_cache_sections: List[torch.Tensor], 
                                  max_length: int) -> List[torch.Tensor]:
        """Truncate kv_cache sections to fit within max_length."""
        truncated_sections = []
        current_pos = 0
        
        for section in kv_cache_sections:
            section_length = section.size(1)
            remaining_length = max_length - current_pos
            
            if remaining_length <= 0:
                break
            elif remaining_length >= section_length:
                truncated_sections.append(section)
                current_pos += section_length
            else:
                truncated_section = section[:, :remaining_length]
                truncated_sections.append(truncated_section)
                break
        
        return truncated_sections

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main collation function with improved logic.
        
        Args:
            features: List of feature dictionaries from dataset
            
        Returns:
            Batched and padded output dictionary
        """
        if not features:
            return {}
        
        # Step 1: Normalize input format for all features
        normalized_features = [self._normalize_input_format(feat) for feat in features]
        
        # Step 2: Split each feature into sections
        all_sections = [self._split_into_sections(feat) for feat in normalized_features]
        
        # Step 3: Pad sections to create uniform batch structure
        output = self._pad_sections(all_sections)
        
        # Step 4: Apply length constraints if needed
        output = self._apply_length_constraints(output)
        
        return output


class BaselineDataCollator:
    """Custom data collator for baseline model training"""
    
    def __init__(self, tokenizer: AutoTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Extract input_ids and labels
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Find max length in batch
        max_length = max(len(ids) for ids in input_ids)
        
        # Apply pad_to_multiple_of if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad sequences
        batch_input_ids = []
        batch_labels = []
        batch_attention_mask = []
        
        for ids, lbls in zip(input_ids, labels):
            # Pad input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * (max_length - len(ids))
            batch_input_ids.append(padded_ids)
            
            # Pad labels (use -100 for padding)
            padded_labels = lbls + [-100] * (max_length - len(lbls))
            batch_labels.append(padded_labels)
            
            # Create attention mask
            attention_mask = [1] * len(ids) + [0] * (max_length - len(ids))
            batch_attention_mask.append(attention_mask)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        }



"""
Helper functions
"""


def create_dataset(dataset_type: str, **kwargs) -> Dataset:
    """
    Factory function to create a dataset based on type.
    
    Args:
        dataset_type: String indicating the type of dataset
        **kwargs: Additional arguments to pass to the dataset constructor
        
    Returns:
        An instance of the appropriate dataset
    """
    # First, check if dataset_type is directly in the registry (exact match)
    if dataset_type in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_type](**kwargs)
    
    # Then check for case-insensitive match
    dataset_type_lower = dataset_type.lower()
    if dataset_type_lower in DATASET_REGISTRY:
        return DATASET_REGISTRY[dataset_type_lower](**kwargs)
    
    # If not found in registry, raise an error with valid options
    valid_options = list(
        set([name for name, cls in DATASET_REGISTRY.items() if name == cls.__name__])
    )  # Only include actual class names
    raise ValueError(
        f"Unknown dataset type: {dataset_type}. Valid options are: {valid_options}"
    )