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
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    load_in_4bit=True,
    device_map="auto"
)

# Set pad_token_id to eos_token_id for open-ended generation
model.config.pad_token_id = model.config.eos_token_id

# Initialize the pipeline for text generation
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Instructions in different languages
instructions = {
    'EN': "Answer the next question like a normal multiple-choice questionnaire and choose the letter of the correct answer.",
    'FR': "Répondez à la question suivante comme dans un questionnaire à choix multiples et choisissez la lettre de la réponse correcte.",
    'DE': "Beantworten Sie die nächste Frage wie in einem normalen Multiple-Choice-Fragebogen und wählen Sie den Buchstaben der richtigen Antwort.",
    'TR': "Bir sonraki soruyu normal bir çoktan seçmeli anket gibi cevaplayın ve doğru cevabın harfini seçin.",
    'JA': "次の質問に、通常の選択式アンケートのように答え、正しい答えの文字を選んでください。"
    # Add more languages if needed
}

def generate_answer(question, options, pipe, language):
    try:
        # Get the instruction based on language
        instruction = instructions[language]
        
        # Format the question with the instruction and options
        formatted_options = ' '.join([f"{key}: {value}" for key, value in eval(options).items()])
        prompt = f"{instruction} {question} Options: {formatted_options}"
        
        # Generate text with the pipeline
        start_time = time.time()  # Start timing
        outputs = pipe(prompt, max_new_tokens=25, num_return_sequences=1)
        end_time = time.time()  # End timing
        generated_text = outputs[0]['generated_text']
        
        # Extract the answer by removing the prompt
        answer = generated_text.replace(prompt, '').strip()
        duration = end_time - start_time  # Calculate duration
        return answer, duration
    except Exception as e:
        return f"Error: {str(e)}", 0

def compare_answers(model_answer, correct_letter):
    try:
        # Extract the chosen letter and name from the model's response
        model_answer = model_answer.lower()
        correct_letter = correct_letter.strip().lower()
        valid_letters = [f"{correct_letter}.", f"{correct_letter}:"]
        
        # Check if the model's chosen letter matches the correct letter
        return 1 if any(letter in model_answer for letter in valid_letters) else 0
    except AttributeError as e:
        print(f"AttributeError: {str(e)} - Likely caused by non-string value in options.")
        return 0


def evaluate_questions(dataframe, start_idx, end_idx):
    results = {'Question': [], 'Options': [], 'Model_Answer': [], 'Correct_Letter': [], 'Compared_Answer': [], 'Time_Taken': [], 'Language': []}
    subset = dataframe.iloc[start_idx:end_idx]
    
    for idx, row in subset.iterrows():
        # Formulate the question
        full_question = row['question']
        options = row['options']
        language = row['language']
        
        # Generate the model's answer
        model_answer, time_taken = generate_answer(full_question, options, pipe, language)
        
        # Compare answers
        compared_answer = compare_answers(model_answer, row['correct_letter'])
        
        # Print answer for immediate feedback
        print(f"Asking: {full_question}")
        print(f"Options: {options}")
        print(f"Correct Letter: {row['correct_letter']}")
        print(f"Model's Response: {model_answer}")
        print(f"Time taken: {time_taken:.2f} seconds")
        print(f"Correct: {compared_answer}")
        
        # Store results
        results['Question'].append(full_question)
        results['Options'].append(options)
        results['Model_Answer'].append(model_answer)
        results['Correct_Letter'].append(row['correct_letter'])
        results['Compared_Answer'].append(compared_answer)
        results['Time_Taken'].append(time_taken)
        results['Language'].append(language)
    
    return pd.DataFrame(results)



# Load the dataset
df = pd.read_csv("all_questions_tr_mc.csv")

# Define the start and end indices for the subset
start_idx = 12119  # Change this value for each run
end_idx = 14632  # Change this value for each run

# Evaluate the questions in the subset of the dataset
results_df = evaluate_questions(df, start_idx, end_idx)

# Print first few results to check
print(results_df.head())

# Optionally, save the results to a CSV file for further analysis
results_df.to_csv(f"all_questions_mc_mistral_tr_results_{start_idx}_{end_idx}.csv", index=False)
results_df.to_json(f'mistral_json_mc_tr_{start_idx}_{end_idx}.json', index=False)
