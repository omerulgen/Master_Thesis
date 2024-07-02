import os
import pandas as pd
import time
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Set the HF token as an environment variable
os.environ['HF_TOKEN'] = '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'

# Set logging level to ERROR to suppress unwanted log messages
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize the tokenizer and model for causal language modeling with 4-bit quantization
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    load_in_4bit=True,
    device_map="auto"
)

# Set pad_token_id to eos_token_id for open-ended generation
model.config.pad_token_id = model.config.eos_token_id

# Initialize the pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Instructions in different languages
instructions = {
    'EN': "Just answer the next statement with True or False.",
    'FR': "Répondez simplement à la prochaine affirmation par Vrai ou Faux.",
    'DE': "Beantworten Sie die nächste Aussage einfach mit Wahr oder Falsch.",
    'TR': "Bir sonraki ifadeyi doğru veya yanlış olarak cevaplayın.",
    'JA': "次の記述に「真」または「偽」で答えてください。"
    # Add more languages if needed
}

def generate_answer(question, pipe, language):
    try:
        # Get the instruction based on language
        instruction = instructions[language]
        
        # Format the question with the instruction
        prompt = f"{instruction} {question}"
        
        # Generate text with the pipeline
        start_time = time.time()  # Start timing
        outputs = pipe(prompt, max_new_tokens=10, num_return_sequences=1)
        end_time = time.time()  # End timing
        generated_text = outputs[0]['generated_text']
        
        # Extract the answer by removing the prompt
        answer = generated_text.replace(prompt, '').strip()
        duration = end_time - start_time  # Calculate duration
        return answer, duration
    except Exception as e:
        return f"Error: {str(e)}", 0

def compare_answers(model_answer, actual_answer):
    # Convert actual_answer to string and normalize to lowercase
    actual_answer = str(actual_answer).lower()
    
    # Normalize model_answer to lowercase
    model_answer = model_answer.lower()
    
    # Define True and False in different languages
    true_values = ["true", "vrai", "wahr", "doğru", "真"]
    false_values = ["false", "faux", "falsch", "yanlış", "偽"]

    # Check if the model's answer matches the actual answer
    if any(true in model_answer for true in true_values) and actual_answer == "true":
        return 1
    elif any(false in model_answer for false in false_values) and actual_answer == "false":
        return 1
    else:
        return 0

def evaluate_questions(dataframe, start_idx, end_idx):
    results = {'Question': [], 'Model_Answer': [], 'Actual_Answer': [], 'Compared_Answer': [], 'Language': [], 'Time_Taken': []}
    subset = dataframe.iloc[start_idx:end_idx]
    
    for idx, row in subset.iterrows():
        # Formulate the question
        full_question = row['question']
        actual_answer = row['True/False']
        language = row['Language']

        # Generate the model's answer
        model_answer, time_taken = generate_answer(full_question, pipe, language)
        
        # Print answer for immediate feedback
        print(f"Asking: {full_question}")
        print(f"Actual Answer: {actual_answer}")
        print(f"Model's Response: {model_answer}")
        print(f"Time taken: {time_taken:.2f} seconds")
        
        # Compare answers
        compared_answer = compare_answers(model_answer, actual_answer)
        
        # Store results
        results['Question'].append(full_question)
        results['Model_Answer'].append(model_answer)
        results['Actual_Answer'].append(actual_answer)
        results['Compared_Answer'].append(compared_answer)
        results['Language'].append(language)
        results['Time_Taken'].append(time_taken)
    
    return pd.DataFrame(results)

# Load the dataset
df = pd.read_csv("true_false_questions_de.csv")

# Define the start and end indices for the subset
start_idx = 0  # Change this value for each run
end_idx = 2288  # Change this value for each run

# Evaluate the questions in the subset of the dataset
results_df = evaluate_questions(df, start_idx, end_idx)

# Print first few results to check
print(results_df.head())

# Optionally, save the results to a CSV file for further analysis
results_df.to_csv(f"true_false_questions_de_{start_idx}_{end_idx}.csv", index=False)
results_df.to_json(f"llama_json_tf_de_{start_idx}_{end_idx}.json", orient='records', lines=True)
