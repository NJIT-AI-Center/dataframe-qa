import os
import argparse
from tqdm import tqdm

import csv
import traceback
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

from utility import (
    load_wikisql,
    get_prompt,
    lowercase_if_string,
    get_llama_prompt,
    get_llama_generated_content,
    get_openai_generated_content,
    extract_code,
    execute_sandboxed_code,
    append_to_csv
)

lowercase_cell_string = True

def main():
    parser = argparse.ArgumentParser(description='Run DataFrame QA on wikisql.')
    parser.add_argument('--model', type=str, default="llama2-7b", help='Name of the LLM')

    args = parser.parse_args()
    model = args.model
    dataset = 'wikisql'
    DATA_PATH = "../data"
    OUTPUT_PATH = "../results"

    output_dir = OUTPUT_PATH + f"/{dataset}/{model}.csv"
    data_dir = DATA_PATH + f'/json/{dataset}.json'

    # Constants
    if model == "llama2-7b":
        model_name = "meta-llama/Llama-2-7b-chat-hf"
    elif model == "llama2-13b":
        model_name = "meta-llama/Llama-2-13b-chat-hf"
    elif model == "llama2-70b":
        model_name = "meta-llama/Llama-2-70b-chat-hf"
    elif model == "codellama-7b":
        model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
    elif model == "codellama-13b":
        model_name = "meta-llama/CodeLlama-13b-Instruct-hf"
    elif model == "codellama-34b":
        model_name = "meta-llama/CodeLlama-34b-Instruct-hf"
    elif model == "gpt3.5":
        model_name = "gpt-3.5-turbo"
    elif model == "gpt4":
        model_name = "gpt-4"

    # Load the dataset
    roles, questions, csv_paths, answers = load_wikisql(data_dir)
    N_SAMPLE = len(questions)

    if model not in ["gpt3.5", "gpt4"]: 
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Initialize the CSV with headers
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Question", "Model Output", "Query", "Output/error", "DataType", "Answer"])

    for i in tqdm(range(N_SAMPLE)):
        question = questions[i]
        df = pd.read_csv(csv_paths[i])
        answer = ','.join(answers[i])
        
        try:
            # lowercase all string cells in a Pandas DataFrame
            if lowercase_cell_string:
                question = question.lower()
                df = df.apply(lambda x: x.apply(lowercase_if_string))
                answer = answer.lower()

            SYS_PROMPT, USER_PROMPT = get_prompt(df, question, lowercase_cell_string=lowercase_cell_string)
            if model not in ["gpt3.5", "gpt4"]: 
                PROMPT = get_llama_prompt(SYS_PROMPT, USER_PROMPT)
                content = get_llama_generated_content(model, tokenizer, PROMPT)
            else:
                content = get_openai_generated_content(model_name, SYS_PROMPT, USER_PROMPT)
            query = extract_code(content)
            result, error = execute_sandboxed_code(query, df)
            
            if error:
                result = f"<ERROR>: {result}"

            datatype = type(result)
            append_to_csv(output_dir, i, question, content, query, result, datatype, answer)

        except Exception as e:
            error_msg = traceback.format_exc()
            append_to_csv(output_dir, i, question, f"<EXCEPTION>: {error_msg}", "", "", "", answer)
        

if __name__ == "__main__":
    main()

