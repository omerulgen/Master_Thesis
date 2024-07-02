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
    'EN': "Just answer the next question in English and do not explain your answer.",
    'FR': "Répondez simplement à la question suivante en français et n'expliquez pas votre réponse",
    'DE': "Beantworten Sie die nächste Frage einfach auf Deutsch und erklären Sie Ihre Antwort nicht.",
    'TR': "Bir sonraki soruyu Türkçe cevaplayın ve cevabınızı açıklamayın.",
    'JA': "次の質問に日本語で答えるだけで、答えを説明する必要はありません。"
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
        outputs = pipe(prompt, max_new_tokens=45, num_return_sequences=1)
        end_time = time.time()  # End timing
        generated_text = outputs[0]['generated_text']
        
        # Extract the answer by removing the prompt
        answer = generated_text.replace(prompt, '').strip()
        duration = end_time - start_time  # Calculate duration
        return answer, duration
    except Exception as e:
        return f"Error: {str(e)}", 0

def compare_answers(model_answer, correct_answer):
    # Handle specific case for numeric answers (like birth years)
    if correct_answer.replace('.0', '') in model_answer:
        return 1
    
    # Split correct answer into words
    correct_words = correct_answer.split()
    
    # If the correct answer has only one word, compare it directly
    if len(correct_words) == 1:
        return 1 if correct_words[0].lower() in model_answer.lower() else 0
    
    # For multi-word answers, check if any significant part is in the model's answer
    for word in correct_words:
        if len(word) > 4 and word.lower() in model_answer.lower():
            return 1
    return 0

def evaluate_questions(dataframe, start_idx, end_idx):
    results = {'Question': [], 'Model_Answer': [], 'Correct_Answer': [], 'Compared_answer': [], 'Language': [], 'Time_Taken': []}
    subset = dataframe.iloc[start_idx:end_idx]
    
    for idx, row in subset.iterrows():
        # Formulate the question
        full_question = row['question']
        language = row['language']

        # Generate the model's answer
        model_answer, time_taken = generate_answer(full_question, pipe, language)
        
        # Print answer for immediate feedback
        print(f"Asking: {full_question}")
        print(f"Correct Answer: {row['correct_answer']}")
        print(f"Model's Response: {model_answer}")
        print(f"Time taken: {time_taken:.2f} seconds")
        
        # Handle specific case for birth year questions
        correct_answer = row['correct_answer']
        if correct_answer.endswith('.0'):
            correct_answer = str(int(float(correct_answer)))
        
        # Compare answers
        compared_answer = compare_answers(model_answer, correct_answer)
        
        # Store results
        results['Question'].append(full_question)
        results['Model_Answer'].append(model_answer)
        results['Correct_Answer'].append(row['correct_answer'])
        results['Compared_answer'].append(compared_answer)
        results['Language'].append(language)
        results['Time_Taken'].append(time_taken)
    
    return pd.DataFrame(results)

# Load the dataset
df = pd.read_csv("arts_openended_clean.csv")

# Define the start and end indices for the subset
start_idx = 15401  # Change this value for each run
end_idx = 22833  # Change this value for each run

# Evaluate the questions in the subset of the dataset
results_df = evaluate_questions(df, start_idx, end_idx)

# Print first few results to check
print(results_df.head())

# Optionally, save the results to a CSV file for further analysis
results_df.to_csv(f"llama_arts_openended_results_{start_idx}_{end_idx}.csv", index=False)
results_df.to_json(f'llama_arts_openended_results_{start_idx}_{end_idx}.json', index=False)
