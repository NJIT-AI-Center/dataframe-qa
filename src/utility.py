import json
import numpy as np
import pandas as pd
import torch
import re
import csv

from openai import OpenAI

# ------------------------
# UTILITY FUNCTIONS
# ------------------------

def load_wikisql(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)
    questions = [item['question'] for item in data]
    csv_paths = [item['csv_filename'] for item in data]
    answers = [item['answer'] for item in data]

    return questions, csv_paths, answers

def load_uci_dataframeqa(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)
    roles = [item['role'] for item in data]
    questions = [item['question'] for item in data]
    csv_paths = [item['csv_filename'] for item in data]
    answer_querys = [item['answer query'] for item in data]

    return roles, questions, csv_paths, answer_querys

def lowercase_if_string(x):
    if isinstance(x, str):
        return x.lower()
    return x


# ------------------------
# MODEL-RELATED FUNCTIONS
# ------------------------

def get_prompt(df, question, lowercase_cell_string=False):
    
    if lowercase_cell_string:
        SYS_PROMPT = """You are a professional Python programming assistant. Write Pandas code to get the answer to the user's question.
- Assumptions: 
  - The Pandas library has been imported as `pd`. You can reference it directly.
  - The dataframe `df` is loaded and available for use.
  - All string values in the `df` have been converted to lowercase.
- Requirements:
  - Use only Pandas operations for the solution.
  - Store the answer in a variable named `result`.
  - Do NOT include comments or explanations in your code.
  - Place your code between the [PYTHON] and [/PYTHON] tags."""
    else:
        SYS_PROMPT = """You are a professional Python programming assistant. Write Pandas code to get the answer to the user's question.
- Assumptions: 
  - The Pandas library has been imported as `pd`. You can reference it directly.
  - The dataframe `df` is loaded and available for use.
- Requirements:
  - Use only Pandas operations for the solution.
  - Store the answer in a variable named `result`.
  - Do NOT include comments or explanations in your code.
  - Place your code between the [PYTHON] and [/PYTHON] tags."""
        
    USER_PROMPT = """You are given a Pandas dataframe named `df`:
- Columns: {}
- Data Types: {}
- User's Question: {}""".format(df.columns.tolist(), get_simplified_dtypes(df), question)

    return SYS_PROMPT, USER_PROMPT


# https://huggingface.co/blog/llama2
def get_llama_prompt(SYS_PROMPT="", USER_PROMPT=""):
    """
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
    """
    prompt = """[INST] 
<<SYS>> 
{}
<</SYS>>

{}
[/INST]""".format(SYS_PROMPT, USER_PROMPT)
    return prompt


def get_openai_generated_content(model="gpt-3.5-turbo", SYS_PROMPT="", USER_PROMPT=""):
    """
    Fetch the model's completion based on system and user prompts.

    Parameters:
    - SYS_PROMPT (str): The system prompt.
    - USER_PROMPT (str): The user prompt.

    Returns:
    - str: The model's generated content.
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
    )

    return completion.choices[0].message.content

def get_llama_generated_content(model, tokenizer, prompt):
    """
    Generate Python code based on the given prompt using the CodeLlama model.
    
    Parameters:
    - model: The trained model to use for generation.
    - tokenizer: The tokenizer to use for encoding/decoding.
    - prompt (str): The instruction prompt for the code generation.

    Returns:
    - str: The generated Python code.
    """
    # Tokenize the prompt and send to appropriate device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda:0')

    # Generate code using the model
    generated_ids = model.generate(
        input_ids, 
        do_sample=False,
        max_new_tokens=500, 
        temperature=0.0,
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Only get the generated portion of the output (i.e., excluding the prompt)
    generated_output_ids = generated_ids[0][input_ids.shape[1]:]
    
    # Decode the generated code and return
    # strip() remove the space at beginning of [PYTHON]
    return tokenizer.decode(generated_output_ids, skip_special_tokens=True).strip()


# Extract code between Python tags
def extract_code(content: dict) -> str:    
    # Regular expression to capture content between [PYTHON] and [/PYTHON]
    pattern = r'\[PYTHON\]\s?(.*?)\s?\[/PYTHON\]'
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else None


# ------------------------
# EXECUTION AND SAFETY
# ------------------------

def execute_sandboxed_code(code_str, df):
    # Define safe built-ins list and other allowed functions/modules
    safe_builtins = {
        'print': print,
        'len': len,
        'range': range,
        'list': list,
        'dict': dict,
        'set': set,
        'int': int,
        'float': float,
        'str': str,
        # Add any other safe built-ins if necessary
    }

    # Define an environment for the execution
    environment = {
        '__builtins__': safe_builtins, 
        'df': df, 
        'pd': pd, 
        'np': np  # including numpy for operations that might require it
    }

    try:
        # Execute code within the restricted environment
        exec(code_str, environment)
    except Exception as e:
        # If an error occurs, return the error message
        return str(e), True

    # Return the result from the environment if available
    return environment.get('result', None), False



def clean_name(name):
    # Remove non-alphanumeric characters (except underscores) and replace spaces with underscores
    cleaned_name = re.sub('[^A-Za-z0-9_]+', '_', name)
    # Remove leading digits
    cleaned_name = re.sub('^[0-9]+', '', cleaned_name)
    return cleaned_name

# def extract_table(raw_data):
#     # Clean the header names
#     cleaned_headers = [clean_name(header) for header in raw_data["header"]]
    
#     # Update the data dictionary with cleaned headers
#     cleaned_data = {
#         "header": cleaned_headers,
#         "rows": raw_data["rows"],
#         "name": clean_name(raw_data["name"])
#     }

#     return cleaned_data

def table_to_df(sql_table):
    df = pd.DataFrame(sql_table['rows'], columns=sql_table['table_header'])
    return df

def string2numeric(df):
    # Check the data types of each column
    data_types = df.dtypes

    # Iterate through the columns and convert them to numeric if they contain numeric data
    for column_name, data_type in data_types.iteritems():
        if data_type == 'object':  # Check if the column contains object data (usually strings)
            try:
                df[column_name] = pd.to_numeric(df[column_name])
            except ValueError:
                # If conversion to numeric fails, handle the error here
                pass

    # Return the DataFrame with converted columns
    return df

# Check for columns with mixed data types
def check_mixed_data_type(df):
    mixed_type_columns = df.columns[df.map(type).nunique() > 1]
    
    # Display the unique data types in each of the problematic columns
    for column in mixed_type_columns:
        unique_types = df[column].apply(type).unique()
        print(f"Column '{column}' has mixed data types: {unique_types}")

    if len(mixed_type_columns) == 0:
        print("No mixed type of columns found")

import ast

# # Convert string "['hello', '0']" to list ['hello', '0']
# # all elements are converted to strings
# def string_to_list(input_string):
#     result_list = ast.literal_eval(input_string)
#     return result_list

def string_to_list(input_string):
    # Replacing nan with 'None' before evaluation
    sanitized_string = input_string.replace('nan', 'None')
    
    try:
        result_list = ast.literal_eval(sanitized_string)
        return result_list
    except ValueError as e:
        print(f"Error for sanitized_string: {sanitized_string}")
        raise e

# Extract sql table from csv data
def csv_to_sql_table(data):
    table_header = string_to_list(data['table_header'])
    table_header_types = string_to_list(data['table_header_types'])
    rows = string_to_list(data['rows'])
    df = pd.DataFrame(rows, columns=table_header)
    
    n_col = len(table_header)
    for i in range(n_col):
        if table_header_types[i] == 'real':
            # Convert column to appropriate numeric type
            df[table_header[i]] = pd.to_numeric(df[table_header[i]])
            
    table = {
        "table_header": string_to_list(data['table_header']),
        "rows": df.values.tolist(),
        "table_name": 'table',
    }
    return table


# ------------------------
# UTILITY FUNCTIONS
# ------------------------

def string_to_numeric(data):
    table_header = data['table_header']
    table_header_types = data['table_header_types']
    rows = data['rows']
    
    df = pd.DataFrame(rows, columns=table_header)

    # Convert columns to appropriate numeric type using vectorized operations
    real_cols = [col for i, col in enumerate(table_header) if table_header_types[i] == 'real']
    df[real_cols] = df[real_cols].apply(pd.to_numeric, errors='coerce')
    
    return df.values.tolist()

def extract_data(index, df_all, labels, lowercase_cell_string=False):
    """
    Given an index, returns the question, dataframe, and label from df_all.

    Parameters:
    - index (int): The index of the desired row in df_all.
    - df_all (pd.DataFrame): The dataframe containing data rows, table headers, and questions.
    - labels (list): A list of labels corresponding to each row in df_all.

    Returns:
    - tuple: (question, df, label)
    """
    
    row = df_all.iloc[index]
    df = pd.DataFrame(row.rows, columns=row.table_header)
    if lowercase_cell_string:
        # Convert all text in df to lowercase
        df = df.apply(lambda col: col.str.lower() if col.dtype == 'object' else col)
    question = row.question
    label = labels[index]

    return question, df, label

def get_simplified_dtypes(df):
    dtypes_map = {
        np.dtype('O'): 'str',
        np.dtype('int64'): 'int',
        np.dtype('float64'): 'float',
        np.dtype('bool'): 'bool',
        # Add more as needed
    }
    return df.dtypes.map(lambda x: dtypes_map.get(x, str(x))).tolist()

# Adjusted datatype determination
def determine_datatype(value):
    datatype = str(np.dtype(type(value)))
    if "U" in datatype:
        return "str"
    elif datatype == "object":
        if isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "dict"
        # Add more type checks as needed
        else:
            return "object"
    else:
        return datatype



    
# https://github.com/deepseek-ai/DeepSeek-Coder
def get_deepseek_prompt(SYS_PROMPT="", USER_PROMPT=""):
    """
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
['content']
### Response:
['content']
<|EOT|>
### Instruction:
['content']
### Response:
    """
    prompt = """{}
### Instruction:
{}""".format(SYS_PROMPT, USER_PROMPT)
    return prompt





def get_llama_generated_questions(model, tokenizer, prompt, max_new_tokens=500, skip_special_tokens=True):
    """
    Generate Python code based on the given prompt using the CodeLlama model.
    
    Parameters:
    - model: The trained model to use for generation.
    - tokenizer: The tokenizer to use for encoding/decoding.
    - prompt (str): The instruction prompt for the code generation.

    Returns:
    - str: The generated Python code.
    """
    # Tokenize the prompt and send to appropriate device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    # Check if CUDA is available and use 'cuda:0', else use CPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    input_ids = input_ids.to(device)
    
    # Generate code using the model
    generated_ids = model.generate(
        input_ids, 
        do_sample=True,
        temperature=0.8, 
        max_new_tokens=max_new_tokens, 
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Only get the generated portion of the output (i.e., excluding the prompt)
    generated_output_ids = generated_ids[0][input_ids.shape[1]:]
    
    # Decode the generated code and return
    # strip() remove the space at beginning of [PYTHON]
    generated_text = tokenizer.decode(generated_output_ids, skip_special_tokens=skip_special_tokens).strip()
    
    # # Free up memory
    # del generated_ids, input_ids, generated_output_ids
    # torch.cuda.empty_cache()
    
    return generated_text




# ------------------------
# FILE OPERATIONS
# ------------------------

# Define the function to append results to a CSV file
def append_to_csv(filename, index, question, content, code_to_run, answer, datatype, label):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([index, question, content, code_to_run, answer, datatype, label])

def append_to_csv_numeric(filename, index, role, question, model_output, query, output, output_datatype, answer_query, answer, answer_datatype):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([index, role, question, model_output, query, output, output_datatype, answer_query, answer, answer_datatype])

# Define the function to append prompts to a CSV file
def prompt_to_csv(filename, index, prompt):
    with open(filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([index, prompt])
        



"""
Updated on 10/25
"""

def load_questions(filename):
    """
    Load questions from a text file.
    Each question is expected to be on a new line.

    :param filename: Path to the file containing questions.
    :return: A list of questions.
    """
    questions = []
    with open(filename, "r", encoding="utf8") as file:
        for line in file:
            # Strip any leading/trailing whitespace, including newlines
            questions.append(line.strip())
    return questions

def load_json_file(filename):
    with open(filename, 'r', encoding='utf8') as file:
        data = json.load(file)
    return data

def table_to_dataframe(table):
    # Extracting header and rows from the table
    headers = table.get('header', [])
    rows = table.get('rows', [])
    
    # Creating the DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df



def get_num_of_tokens(prompt, tokenizer):
    # Tokenize the prompt and send to appropriate device
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    return len(input_ids[0])
    
def code_debug_prompt(question, df, code, result, label, tokenizer, max_length=3000):
    if len(df) > 100:
        df = df.iloc[:100]
    while True:
        prompt = """- Task Description:
I am trying to process a DataFrame with Python to find out the answer to the question. My code is not producing the expected result, and I need assistance to identify the discrepancy.

- Question:
{}

- DataFrame `df`:
{}

- My code:
{}

- Actual Output of My Code:
{}

- Expected output:
{}

- Request for Help:
    - Analyze why the output from my code doesn’t match the expected result? I'm not sure if the problem lies in the way I'm querying the DataFrame, or if the question is unclear since I don't have access to the data when I write the Python code. I only have the question and the table header. 
    - Help me figure out if I've written the code incorrectly, or if the expected result itself is incorrect?
    - Provide a concise answer.
""".format(question, df.to_string(), code, result, label)
        num_tokens = get_num_of_tokens(prompt, tokenizer)
        if num_tokens <= max_length:
            return prompt
        else:
            # remove a row
            df = df.iloc[:-1]

def get_error_type_prompt(question, df, code, result, label):
    df = df.iloc[:5]
    prompt = """- Task Description:
I am trying to process a DataFrame with Python to find out the answer to the question. My code is not producing the expected result, and I need assistance to identify the error type.

- Question:
{}

- Sample Rows from DataFrame `df`:
{}

- My Code:
{}

- Actual Output of My Code:
{}

- Expected Output:
{}

- Error Class and Description
| Error Class                             | Description |
|-----------------------------------------|-------------|
| Query Condition and Value Errors        | This class covers errors where query conditions do not reflect the data accurately or the wrong values are used, resulting in no matches or incorrect results. It includes using incorrect column names or values and failing to match the query criteria with the dataset. |
| String Matching and Comparison Errors   | Errors in this class arise from improper handling of string comparisons, such as failing to use appropriate matching methods, not accounting for case sensitivity, whitespace, or special characters, and using exact matching where pattern recognition is required. |
| Data Type and Operation Errors          | This class includes errors from attempting operations between incompatible data types, using methods unsuitable for the data type, and applying aggregation functions incorrectly, often leading to type mismatches or operation errors on non-compatible data types. |
| Data Access and Bounds Errors           | This class is for errors when data is accessed using an incorrect index or key, or when the index exceeds the bounds of the data structure, leading to 'index out of bounds' or 'key not found' errors. |
| Expectation and Interpretation Errors   | This class encompasses errors from a discrepancy between expected outcomes and actual results, which may stem from misinterpreting the output, data, or having incorrect expectations of the data's structure, leading to incorrect assumptions and results. |
| Function and Method Usage Errors        | Errors in this category result from misusing functions or methods outside their intended purpose, such as using a function designed for a specific operation in a context where it does not apply, or calling methods on objects they are not designed for. |
| Data Structure Reference Errors         | This class refers to errors arising from incorrect assumptions or references to the data's structure, such as referencing non-existent columns or misinterpreting the content of the data, leading to queries that do not align with the actual data format or content. |
| Others                                  | A category to cover any errors that do not fit into the specific categories above, such as general mistakes in code logic or implementation that leads to unexpected results or errors. |

- Assumption:
    - All the string values in the `df` are converted to lowercase.
    - All the words in the question are converted to lowercase.
    - I can only access the dataframe column names when I write the Python code. I don't have access to the data itself. So my code may not be correct if the question is unclear or the column names are unclear.

- Requirement:
    - Classify the error type in my code from the error class table above. Choose all error classes from above and place your answer between the [ERROR] and [/ERROR] tags. For example: \n[ERROR]\nQuery Condition and Value Errors\n[/ERROR]\n
    - Error could belong to multiple error classes.
    - Check dataset quality, i.e., the question and the column names of the DataFrame to determine if they are clear and accurate. Choose one option from 'Both are clear', 'Question is unclear', 'Column names are unclear', and 'Both are unclear'. Place your answer between the [QUALITY] and [/QUALITY] tags. For example: \n[QUALITY]\nQuestion is unclear\n[/QUALITY]\n
    - If either the question or the column names are unclear, provide a revised version of the unclear component. Place your answer between the [REVISED] and [/REVISED] tags. For example: \n[REVISED]\nQuestion: What is the average age of the students?\n[/REVISED]\n
    - Don't include comments or explanations in your answer.
""".format(question, df.to_csv(), code, result, label)
    return prompt


def get_error_type_prompt2(question, df, code, result, label):
    df = df.iloc[:5]
    prompt = """- Task Description:
I am trying to process a DataFrame with Python to find out the answer to the question. My code is not producing the expected result, and I need assistance to identify the error type.

- Question:
{}

- Sample Rows from DataFrame `df`:
{}

- My Code:
{}

- Actual Output of My Code:
{}

- Expected Output:
{}

- Error Class and Description
| Error Class                             | Description |
|-----------------------------------------|-------------|
| Query Condition and Value Errors        | This class covers errors where query conditions do not reflect the data accurately or the wrong values are used, resulting in no matches or incorrect results. It includes using incorrect column names or values and failing to match the query criteria with the dataset. |
| String Matching and Comparison Errors   | Errors in this class arise from improper handling of string comparisons, such as failing to use appropriate matching methods, not accounting for case sensitivity, whitespace, or special characters, and using exact matching where pattern recognition is required. |
| Data Type and Operation Errors          | This class includes errors from attempting operations between incompatible data types, using methods unsuitable for the data type, and applying aggregation functions incorrectly, often leading to type mismatches or operation errors on non-compatible data types. |
| Data Access and Bounds Errors           | This class is for errors when data is accessed using an incorrect index or key, or when the index exceeds the bounds of the data structure, leading to 'index out of bounds' or 'key not found' errors. |
| Expectation and Interpretation Errors   | This class encompasses errors from a discrepancy between expected outcomes and actual results, which may stem from misinterpreting the output, data, or having incorrect expectations of the data's structure, leading to incorrect assumptions and results. |
| Function and Method Usage Errors        | Errors in this category result from misusing functions or methods outside their intended purpose, such as using a function designed for a specific operation in a context where it does not apply, or calling methods on objects they are not designed for. |
| Data Structure Reference Errors         | This class refers to errors arising from incorrect assumptions or references to the data's structure, such as referencing non-existent columns or misinterpreting the content of the data, leading to queries that do not align with the actual data format or content. |
| Others                                  | A category to cover any errors that do not fit into the specific categories above, such as general mistakes in code logic or implementation that leads to unexpected results or errors. |

- Assumption:
    - All the string values in the `df` are converted to lowercase.
    - All the words in the question are converted to lowercase.
    - I can only access the dataframe column names when I write the Python code. I don't have access to the data itself. So my code may not be correct if the question is unclear or the column names are unclear.

- Requirement:
    - Classify the error type in my code from the error class table above. Choose all error classes from above and place your answer between the [ERROR] and [/ERROR] tags. For example: \n[ERROR]\nQuery Condition and Value Errors\n[/ERROR]\n
    - Error could belong to multiple error classes.
    - Provide a concise justification for your error classification decision. Place your answer between the [EXPLANATION] and [/EXPLANATION] tags. For example: \n[EXPLANATION]\nYour explanation\n[/EXPLANATION]\n
    - Check dataset quality, i.e., the question and the column names of the DataFrame to determine if they are clear and accurate. Choose one option from 'Both are clear', 'Question is unclear', 'Column names are unclear', and 'Both are unclear'. Place your answer between the [QUALITY] and [/QUALITY] tags. For example: \n[QUALITY]\nQuestion is unclear\n[/QUALITY]\n
    - If either the question or the column names are unclear, provide a revised version of the unclear component. Place your answer between the [REVISED] and [/REVISED] tags. For example: \n[REVISED]\nQuestion: What is the average age of the students?\n[/REVISED]\n
    - Don't include comments or explanations in your answer.
""".format(question, df.to_csv(), code, result, label)
    return prompt


def get_error_type_prompt3(question, df, code, result, label):
    df = df.iloc[:5]
    prompt = """- Task Description:
I am trying to process a DataFrame with Python to find out the answer to the question. My code is not producing the expected result. I need you to classify the error. Below in table I have mentioned class names and class description use that information to do classification. Class assigned should be from the below table. Carefully use class description for classification task.

+----+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|    | Class Name                            | Class Description                                                                                                                                                                                                                                                           |
|----+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  0 | Query Condition and Value Errors      | This class covers errors where query conditions do not reflect the data accurately or the wrong values are used, resulting in no matches or incorrect results. It includes using incorrect column names or values and failing to match the query criteria with the dataset. |
|  1 | String Matching and Comparison Errors | Errors in this class arise from improper handling of string comparisons, such as failing to use appropriate matching methods, not accounting for case sensitivity, whitespace, or special characters, and using exact matching where pattern recognition is required.       |
|  2 | Data Type and Operation Errors        | This class includes errors from attempting operations between incompatible data types, using methods unsuitable for the data type, and applying aggregation functions incorrectly, often leading to type mismatches or operation errors on non-compatible data types.       |
|  3 | Data Access and Bounds Errors         | This class is for errors when data is accessed using an incorrect index or key, or when the index exceeds the bounds of the data structure, leading to 'index out of bounds' or 'key not found' errors.                                                                     |
|  4 | Expectation and Interpretation Errors | This class encompasses errors from a discrepancy between expected outcomes and actual results, which may stem from misinterpreting the output, data, or having incorrect expectations of the data's structure, leading to incorrect assumptions and results.                |
|  5 | Function and Method Usage Errors      | Errors in this category result from misusing functions or methods outside their intended purpose, such as using a function designed for a specific operation in a context where it does not apply, or calling methods on objects they are not designed for.                 |
|  6 | Data Structure Reference Errors       | This class refers to errors arising from incorrect assumptions or references to the data's structure, such as referencing non-existent columns or misinterpreting the content of the data, leading to queries that do not align with the actual data format or content.     |
|  7 | Miscellaneous Errors                  | A new category to cover any errors that do not fit into the specific categories above, such as general mistakes in code logic or implementation that leads to unexpected results or errors.                                                                                 |
+----+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

- Question:
{}

- Sample DataFrame `df`:
{}

- Code:
{}

- Error:
{}

- Expected output:
{}

- Additional Details
  - Code is written to solve the question.
  - We are running the code on the dataframe.
  - Error is what we got after executing the code.
  - Expected output is like ground truth for the question.
- Response:
  - let your response be in a python dictionary format.
  - I need only two things one class name and a 1-2 line explanation for choosing class.
- Response format:
{{
  "Class":"Predicted class name",
  "Explanation":"...."
}}
""".format(question, df.to_csv(), code, result, label)
    return prompt


from ucimlrepo import fetch_ucirepo

def get_uci_dataset(id=1):
    """
    Fetches a dataset from the UCI repository.
    
    Returns:
    dict: A dictionary containing dataset metadata and the dataframe.
    """
    dataset = fetch_ucirepo(id=id)
    return {
        "uci_id": dataset['metadata']['uci_id'],
        "name": dataset['metadata']['name'],
        "area": dataset['metadata']['area'],
        "description": dataset['metadata']['additional_info']['summary'],
        "column_dict": dict(zip(dataset['variables']['name'], dataset['variables']['description'])),
        "dataframe": dataset.data.original,
    }

def get_kaggle_dataset(id=1):
    """
    Fetches a dataset from the UCI repository.
    
    Returns:
    dict: A dictionary containing dataset metadata and the dataframe.
    """
    summary = pd.read_csv("../data/kaggle/kaggle_summary.csv")
    dataset = summary.iloc[id-1]
    df = pd.read_csv(f"../data/kaggle/kaggle_df/{id}.csv")
    return {
        "dataset_id": dataset['id'],
        "name": dataset['Name'],
        "description": dataset['Description'],
        "dataframe": df,
    }