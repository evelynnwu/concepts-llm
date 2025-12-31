# concepts-llm

Fine-tuning Llama 3.1 8B to help students with 21-127 Concepts of Mathematics at CMU.

## Project Structure

```
concepts-llm/
├── prepare_data.py      # Convert LaTeX notes to training data
├── llama3.1(8b)         # Unsloth fine-tuning notebook (run in Colab)
├── examples/
│   └── sample_lecture.tex  # Sample LaTeX format
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install pymupdf  # For PDF processing
```

### 2. Prepare Your Training Data

**Option A: Process a PDF textbook (recommended for your use case)**

```bash
# Auto-parse PDF (tries to extract definitions, theorems, proofs, examples)
python prepare_data.py --pdf textbook.pdf --output_file training_data.jsonl

# OR: Export for manual annotation (better quality, more effort)
python prepare_data.py --pdf textbook.pdf --export_for_annotation --output_file chunks.json
# Edit chunks.json to add instruction/output fields, then:
python prepare_data.py --annotated_file chunks.json --output_file training_data.jsonl
```

**Option B: Process LaTeX files**

```bash
# Basic usage - process lecture notes
python prepare_data.py --input_dir ./lecture_notes --output_file training_data.jsonl

# With separate problem sets directory
python prepare_data.py \
    --input_dir ./lecture_notes \
    --problem_sets_dir ./problem_sets \
    --output_file training_data.jsonl
```

### 3. Expected LaTeX Format

The script recognizes these environments (customize in `prepare_data.py` if needed):

- `\begin{definition}[Title]...\end{definition}`
- `\begin{theorem}[Title]...\end{theorem}` (also: lemma, proposition, corollary)
- `\begin{proof}...\end{proof}`
- `\begin{example}...\end{example}`
- `\begin{exercise}...\end{exercise}` (also: problem)
- `\begin{solution}...\end{solution}`

See `examples/sample_lecture.tex` for a complete example.

### 3. Fine-tune the Model

1. Upload `training_data.jsonl` to Google Drive or Hugging Face
2. Open `llama3.1(8b)` notebook in Google Colab
3. Modify the dataset loading cell to use your data:

```python
from datasets import load_dataset
dataset = load_dataset('json', data_files='training_data.jsonl', split='train')
```

4. Run all cells to fine-tune

## Training Data Format

The script generates JSONL with this structure:

```json
{"instruction": "Define the Principle of Mathematical Induction.", "input": "", "output": "Let P(n) be a statement..."}
{"instruction": "Prove that the sum 1+2+...+n = n(n+1)/2", "input": "", "output": "We proceed by induction..."}
```

## Customization

### Adding Custom Environments

Edit `extract_all_items()` in `prepare_data.py`:

```python
env_mappings = {
    'definition': 'definition',
    'theorem': 'theorem',
    'myenv': 'example',  # Add your custom environments
    ...
}
```

### Generating Different Question Types

Modify `generate_instruction_pairs()` to create different types of Q&A pairs from the same content.

## PDF Processing Workflow

Since your notes come as a PDF textbook, here's the recommended workflow:

### Approach 1: Automatic Parsing (Quick Start)
```bash
python prepare_data.py --pdf your_textbook.pdf --output_file training_data.jsonl
```
- Works well if textbook has clear "Definition:", "Theorem:", "Proof:" labels
- Review output and manually fix any issues

### Approach 2: Manual Annotation (Higher Quality)
```bash
# Step 1: Export chunks for annotation
python prepare_data.py --pdf your_textbook.pdf --export_for_annotation

# Step 2: Edit the generated JSON file (training_data.json)
# For each chunk, add:
#   "instruction": "What question should a student ask?"
#   "output": "What should the tutor respond?"

# Step 3: Convert to training format
python prepare_data.py --annotated_file training_data.json --output_file training_data.jsonl
```

### Approach 3: Hybrid (Recommended)
1. Run auto-parsing first to see what it extracts
2. Manually create additional Q&A pairs for:
   - Common student questions from office hours
   - Tricky concepts that need extra explanation
   - Problem-solving strategies
   - Worked examples with detailed steps

## Topics Covered in 21-127

- Mathematical induction (weak and strong)
- Set theory and operations
- Relations (equivalence relations, partial orders)
- Functions (injective, surjective, bijective)
- Combinatorics and counting
- Graph theory basics
- Number theory (divisibility, primes, modular arithmetic)
- Proof techniques

## Tips for Quality Training Data

1. **Include reasoning steps** - Don't just give final answers
2. **Cover common mistakes** - Add examples correcting misconceptions
3. **Vary difficulty** - Include easy, medium, and hard examples
4. **Multiple proof styles** - Show different valid approaches
5. **Add hint-mode examples** - Train model to give hints without full solutions
