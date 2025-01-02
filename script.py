# setup
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer, pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import matplotlib.pyplot as plt


# utility methods
get_yes_no_prompt = lambda context, question, response: f"""
    Context: {context}\n
    Question: {question}\n
    Response: {response}\n
    Based on the given Context and Question, answer this question:\n
    Is the provided Response correct? Answer only Yes or No.\n
    Answer:\n""".strip()

def get_yes_score(outputs, input_length, tokenizer):
    generated_tokens = outputs.sequences[:, input_length:]
    # 1. find the index (idx) of the first character-based token.
    for idx, tok in enumerate(generated_tokens[0]):
        next_token_str = tokenizer.decode(tok, skip_special_tokens=True)
        n_letters = sum(c.isalpha() for c in next_token_str)
        if n_letters != len(next_token_str):
            continue
        break
    # 2a. do preselection on high probabilities (out of 32k tokens)
    probs_all = torch.nn.functional.softmax(outputs.logits[idx][0], dim=-1)
    indices = torch.argwhere(probs_all > 0.001)
    indices = indices[:, -1]
    tokens_max = tokenizer.batch_decode(indices, skip_special_tokens=True)
    probs_max = probs_all[probs_all > 0.001]
    # 2b. find yes/no probabilities
    next_token_dict = {str(t): p for t, p in zip(tokens_max, probs_max)}
    yes_prob = next_token_dict.get("Yes", 0.)
    no_prob = next_token_dict.get("No", 0.)
    # 3. calculate and return yes/no confidence score
    yes_score = yes_prob / (yes_prob + no_prob) if yes_prob != 0 or no_prob != 0 else 0.5
    return yes_score.item() if type(yes_score)==torch.Tensor else yes_score

def plot_histogram(scores, title):
    """Simple wrapper for generating a histogram from the input scores, saving the figure to a pdf based on the title"""
    plt.hist(scores, range=(0, 1.0), bins=50)
    plt.xlabel("Yes Score")
    plt.ylabel("Number of Questions")
    plt.title(title)
    plt.savefig(f"{title}.pdf")
    plt.clf()


# models - @pram
# NOTE: the original experiment uses "meta-llama/Llama-2-13b-chat-hf",
# but here I'm using a smaller model due to GPU constraints of a2-highgpu-1g, which is the hardware available in our GCP instance
device = "cuda" if torch.cuda.is_available() else "cpu"
qa_model_name = "deepset/roberta-base-squad2"
llama_model_name = "meta-llama/Llama-2-7b-chat-hf" 
llama_model = LlamaForCausalLM.from_pretrained(llama_model_name, device_map=device)
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_name, device_map=device)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

# data
NUM_RECORDS = 1000 # @param
data = load_dataset("rajpurkar/squad_v2", split=f"train[:{NUM_RECORDS}]")


# yes-no scoring - ground-truth
ground_truth_scores = []
for i, row in tqdm(enumerate(data)):
    if len(row['answers']['text']) < 1: continue
    prompt = get_yes_no_prompt(row['context'], row['question'], row['answers']['text'][0])
    input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_length = input_ids.shape[1]
    outputs = llama_model.generate(input_ids, output_logits=True, return_dict_in_generate=True, max_new_tokens=5)
    yes_score = get_yes_score(outputs, input_length, llama_tokenizer)
    ground_truth_scores.append(yes_score)

plot_histogram(ground_truth_scores, "Histogram of Yes Scores Using Ground Truth Responses")


# yes-no scoring - qa model
qa_scores = []
for i, row in tqdm(enumerate(data)):
    nlp = pipeline('question-answering', model=qa_model, tokenizer=qa_tokenizer)
    QA_input = {'question': row["question"], 'context': row['context']}
    response = nlp(QA_input)
    prompt = get_yes_no_prompt(row['context'], row['question'], response)
    input_ids = llama_tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    input_length = input_ids.shape[1]
    outputs = llama_model.generate(input_ids, output_logits=True, return_dict_in_generate=True, max_new_tokens=5)
    yes_score = get_yes_score(outputs, input_length, llama_tokenizer)
    qa_scores.append(yes_score)

plot_histogram(qa_scores, f"Histogram of Yes Scores Using Answers from {qa_model_name.split('/')[-1]}")