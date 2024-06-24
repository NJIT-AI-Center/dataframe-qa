import os
import argparse
from tqdm import tqdm

import csv
import traceback
import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM

from utility import (
    load_uci_dataframeqa,
    get_prompt,
    get_llama_prompt,
    get_llama_generated_content,
    get_openai_generated_content,
    extract_code,
    execute_sandboxed_code,
    append_to_csv_numeric
)

lowercase_cell_string = False

def main():
    parser = argparse.ArgumentParser(description='Run DataFrame QA on uci-dataframeqa.')
    parser.add_argument('--model', type=str, default="llama2-7b", help='Name of the LLM')

    args = parser.parse_args()
    model = args.model
    dataset = 'uci_dataframeqa'
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
    roles, questions, csv_paths, answer_querys = load_uci_dataframeqa(data_dir)
    N_SAMPLE = len(questions)

    if model not in ["gpt3.5", "gpt4"]: 
        # Initialize the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

    # Initialize the CSV with headers
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    with open(output_dir, "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Role", "Question", "Model Output", "Query", "Output/error", "DataType", "Answer Query", "Answer", "Answer Datatype"])

    for i in tqdm(range(N_SAMPLE)):  
        role = roles[i]
        question = questions[i]
        df = pd.read_csv(csv_paths[i])
        answer_query = answer_querys[i]
        
        try:
            SYS_PROMPT, USER_PROMPT = get_prompt(df, question, lowercase_cell_string=lowercase_cell_string)
            if model not in ["gpt3.5", "gpt4"]: 
                PROMPT = get_llama_prompt(SYS_PROMPT, USER_PROMPT)
                content = get_llama_generated_content(model, tokenizer, PROMPT)
            else:
                content = get_openai_generated_content(model_name, SYS_PROMPT, USER_PROMPT)
            query = extract_code(content)
            output, error = execute_sandboxed_code(query, df)

            if error:
                output = f"<ERROR>: {output}"
            output_datatype = type(output)

            answer, answer_error = execute_sandboxed_code(answer_query, df)
            if answer_error:
                answer = f"<ERROR>: {answer}"
            answer_datatype = type(answer)

            append_to_csv_numeric(output_dir, i, role, question, content, query, output, output_datatype, answer_query, answer, answer_datatype)

        except Exception as e:
            error_msg = traceback.format_exc()
            append_to_csv_numeric(output_dir, i, role, question, f"<EXCEPTION>: {error_msg}", "", "", "", answer_query, "", "")

if __name__ == "__main__":
    main()