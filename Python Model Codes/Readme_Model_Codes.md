### Models and File Structure

Due to the sheer number of instances, we have split the dataset into six parts for each model to ensure efficient processing. Each file's name indicates the model, the section of the dataset it addresses, the language, and the question format.

#### Example File Names

1. **llama3_code_de_tf.py**
   - **Model**: LLama3-8B-Instruct
   - **Language**: German (de)
   - **Question Format**: True/False (tf)
   - **Dataset Section**: First set of German people

2. **mistral_code_fr_mc4.py**
   - **Model**: MISTRAL-7b-Instruct
   - **Language**: French (fr)
   - **Question Format**: Multiple Choice (mc)
   - **Dataset Section**: Fourth section of the French dataset

### File Naming Convention

Each file name follows the pattern: `<model>_code_<language>_<question_format><dataset_section>.py`

- **Model**: Either `llama3` or `mistral`
- **Language**: `en` (English), `fr` (French), `tr` (Turkish), `de` (German), `ja` (Japanese)
- **Question Format**: `mc` (Multiple Choice), `oe` (Open-Ended), `tf` (True/False)
- **Dataset Section**: Indicates which part of the dataset the file handles (e.g., `1`, `2`, `3`, `4`, `5`, `6`)

### File Examples

- **LLama3 Model**:
  - `llama3_code_en_mc1.py`: Handles the first section of the English dataset for Multiple Choice questions.
  - `llama3_code_ja_oe3.py`: Handles the third section of the Japanese dataset for Open-Ended questions.
  
- **MISTRAL Model**:
  - `mistral_code_tr_tf2.py`: Handles the second section of the Turkish dataset for True/False questions.
  - `mistral_code_de_mc5.py`: Handles the fifth section of the German dataset for Multiple Choice questions.
