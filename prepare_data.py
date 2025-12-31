#!/usr/bin/env python3
"""
Data preparation script for converting course materials (PDF or LaTeX)
into fine-tuning training data for a discrete math tutor.

Usage:
    # For PDF textbooks
    python prepare_data.py --pdf textbook.pdf --output_file training_data.jsonl

    # For LaTeX files
    python prepare_data.py --input_dir ./latex_notes --output_file training_data.jsonl
"""

import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# PDF processing imports (install with: pip install pymupdf)
try:
    import fitz  # PyMuPDF
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False


@dataclass
class ExtractedItem:
    """Represents an extracted piece of content from LaTeX."""
    item_type: str  # definition, theorem, proof, example, exercise
    title: Optional[str]
    content: str
    label: Optional[str] = None
    associated_proof: Optional[str] = None


def clean_latex(text: str) -> str:
    """Clean LaTeX markup while preserving mathematical content."""
    # Remove comments
    text = re.sub(r'(?<!\\)%.*$', '', text, flags=re.MULTILINE)

    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    # Keep math environments readable
    # Convert \[ \] to $$ $$
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)

    # Remove some common formatting commands but keep content
    text = re.sub(r'\\textbf\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\textit\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\emph\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\underline\{([^}]*)\}', r'\1', text)

    # Clean up labels and refs (keep refs readable)
    text = re.sub(r'\\label\{[^}]*\}', '', text)
    text = re.sub(r'\\ref\{([^}]*)\}', r'[\1]', text)

    return text


def extract_environment(text: str, env_name: str) -> list[tuple[str, str, str]]:
    """
    Extract all instances of a LaTeX environment.

    Returns list of (title, content, label) tuples.
    """
    # Pattern matches \begin{env}[optional title] ... \end{env}
    # Also captures optional \label{...} inside
    pattern = rf'\\begin\{{{env_name}\}}(?:\[([^\]]*)\])?\s*(.*?)\\end\{{{env_name}\}}'
    matches = re.findall(pattern, text, re.DOTALL)

    results = []
    for match in matches:
        title = match[0] if match[0] else None
        content = match[1].strip()

        # Extract label if present
        label_match = re.search(r'\\label\{([^}]*)\}', content)
        label = label_match.group(1) if label_match else None

        # Clean the content
        content = clean_latex(content)

        results.append((title, content, label))

    return results


def extract_all_items(latex_content: str) -> list[ExtractedItem]:
    """Extract all relevant items from LaTeX content."""
    items = []

    # Common environment names (customize based on your notes)
    env_mappings = {
        'definition': 'definition',
        'defn': 'definition',
        'def': 'definition',
        'theorem': 'theorem',
        'thm': 'theorem',
        'lemma': 'lemma',
        'proposition': 'proposition',
        'prop': 'proposition',
        'corollary': 'corollary',
        'cor': 'corollary',
        'example': 'example',
        'ex': 'example',
        'exercise': 'exercise',
        'problem': 'exercise',
        'proof': 'proof',
        'solution': 'solution',
        'remark': 'remark',
    }

    for env_name, item_type in env_mappings.items():
        extractions = extract_environment(latex_content, env_name)
        for title, content, label in extractions:
            if content:  # Skip empty environments
                items.append(ExtractedItem(
                    item_type=item_type,
                    title=title,
                    content=content,
                    label=label
                ))

    return items


def pair_theorems_with_proofs(items: list[ExtractedItem]) -> list[ExtractedItem]:
    """Associate proofs with their preceding theorems/lemmas."""
    paired_items = []
    pending_theorem = None

    for item in items:
        if item.item_type in ('theorem', 'lemma', 'proposition', 'corollary'):
            if pending_theorem:
                paired_items.append(pending_theorem)
            pending_theorem = item
        elif item.item_type == 'proof' and pending_theorem:
            pending_theorem.associated_proof = item.content
            paired_items.append(pending_theorem)
            pending_theorem = None
        else:
            if pending_theorem:
                paired_items.append(pending_theorem)
                pending_theorem = None
            if item.item_type != 'proof':  # Don't add orphan proofs
                paired_items.append(item)

    if pending_theorem:
        paired_items.append(pending_theorem)

    return paired_items


def generate_instruction_pairs(item: ExtractedItem) -> list[dict]:
    """Convert an extracted item into instruction-response training pairs."""
    pairs = []

    if item.item_type == 'definition':
        # Generate "what is" questions
        if item.title:
            pairs.append({
                "instruction": f"Define {item.title} in discrete mathematics.",
                "input": "",
                "output": item.content
            })
            pairs.append({
                "instruction": f"What is {item.title}?",
                "input": "",
                "output": item.content
            })
        else:
            pairs.append({
                "instruction": "Explain the following definition.",
                "input": "",
                "output": item.content
            })

    elif item.item_type in ('theorem', 'lemma', 'proposition', 'corollary'):
        theorem_name = item.title if item.title else f"this {item.item_type}"

        if item.associated_proof:
            # Full proof request
            pairs.append({
                "instruction": f"State and prove {theorem_name}.",
                "input": "",
                "output": f"**{item.item_type.capitalize()}:** {item.content}\n\n**Proof:** {item.associated_proof}"
            })
            # Just proof request
            pairs.append({
                "instruction": f"Prove that {item.content}",
                "input": "",
                "output": item.associated_proof
            })
        else:
            # Just state the theorem
            pairs.append({
                "instruction": f"State {theorem_name}.",
                "input": "",
                "output": item.content
            })

    elif item.item_type == 'example':
        pairs.append({
            "instruction": "Work through the following example step by step.",
            "input": item.title if item.title else "",
            "output": item.content
        })

    elif item.item_type == 'exercise':
        if item.associated_proof:  # Has solution
            pairs.append({
                "instruction": item.content,
                "input": "",
                "output": item.associated_proof
            })
        # Also generate hint version
        pairs.append({
            "instruction": f"Give me a hint for solving: {item.content}",
            "input": "",
            "output": generate_hint(item)
        })

    elif item.item_type == 'remark':
        if item.title:
            pairs.append({
                "instruction": f"Explain the significance of {item.title}.",
                "input": "",
                "output": item.content
            })

    elif item.item_type == 'note':
        # Notes often contain important clarifications and examples
        pairs.append({
            "instruction": "Explain this important note about the concept.",
            "input": "",
            "output": item.content
        })
        # If the note contains an example with sets, generate a more specific question
        if 'example' in item.content.lower() or '=' in item.content:
            pairs.append({
                "instruction": "Give me an example that illustrates this concept.",
                "input": "",
                "output": item.content
            })

    return pairs


def generate_hint(item: ExtractedItem) -> str:
    """Generate a hint for an exercise (placeholder - customize based on content)."""
    # This is a simple heuristic - you may want to manually curate hints
    content = item.content.lower()

    if 'induction' in content or 'for all n' in content:
        return "Consider using mathematical induction. What would be your base case, and what do you need to show in the inductive step?"
    elif 'prove' in content and 'if' in content:
        return "Think about what proof technique would work best here: direct proof, contrapositive, or contradiction?"
    elif 'count' in content or 'how many' in content:
        return "Is this a permutation or combination problem? Consider whether order matters and whether repetition is allowed."
    elif 'set' in content:
        return "Try to express this using set notation and apply the relevant set operations or identities."
    else:
        return "Start by carefully identifying what you're given and what you need to show. What definitions or theorems might be relevant?"


def process_latex_file(filepath: Path) -> list[dict]:
    """Process a single LaTeX file and return training pairs."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract items
    items = extract_all_items(content)

    # Pair theorems with proofs
    items = pair_theorems_with_proofs(items)

    # Generate training pairs
    all_pairs = []
    for item in items:
        pairs = generate_instruction_pairs(item)
        for pair in pairs:
            pair['source_file'] = str(filepath.name)
        all_pairs.extend(pairs)

    return all_pairs


def process_problem_set(filepath: Path) -> list[dict]:
    """
    Process a problem set with solutions.

    Expects format like:
    \begin{problem}
    ...
    \end{problem}
    \begin{solution}
    ...
    \end{solution}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    pairs = []

    # Extract problems and solutions
    problems = extract_environment(content, 'problem')
    solutions = extract_environment(content, 'solution')

    # Also try 'exercise' environment
    problems.extend(extract_environment(content, 'exercise'))

    # Pair them up (assuming they appear in order)
    for i, (title, problem_content, label) in enumerate(problems):
        pair = {
            "instruction": f"Solve the following problem:\n\n{problem_content}",
            "input": "",
            "output": "",
            "source_file": str(filepath.name)
        }

        if i < len(solutions):
            _, solution_content, _ = solutions[i]
            pair["output"] = solution_content

        if pair["output"]:  # Only add if we have a solution
            pairs.append(pair)

    return pairs


# =============================================================================
# PDF Processing Functions
# =============================================================================

def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    if not HAS_PDF_SUPPORT:
        raise ImportError("PyMuPDF not installed. Run: pip install pymupdf")

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n\n"
    doc.close()
    return text


def parse_pdf_sections(text: str) -> list[dict]:
    """
    Parse PDF text into sections based on common textbook patterns.
    Returns a list of sections with their type and content.

    Optimized for CMU 21-127 Concepts textbook format:
    - "Definition." followed by content (in pink boxes)
    - "Theorem X.Y.Z (Name)." followed by content (in green boxes)
    - "Note." for explanatory remarks
    - "Proof." for proofs
    """
    sections = []

    # Patterns optimized for CMU Concepts textbook format
    patterns = {
        # Definition: "Definition." followed by content until next section
        'definition': [
            r'Definition\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Lemma\s+\d|Proposition\s+\d|Corollary\s+\d|Note\.|Proof\.|Example\s*\d|\Z))',
        ],
        # Theorem: "Theorem X.Y.Z (Name)." followed by content
        'theorem': [
            r'Theorem\s+([\d.]+)\s*\(([^)]+)\)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Lemma\s+\d|Note\.|Proof\.|Example|\Z))',
            r'Theorem\s+([\d.]+)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Lemma\s+\d|Note\.|Proof\.|Example|\Z))',
        ],
        # Lemma patterns
        'lemma': [
            r'Lemma\s+([\d.]+)\s*\(([^)]+)\)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Lemma\s+\d|Note\.|Proof\.|\Z))',
            r'Lemma\s+([\d.]+)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Lemma\s+\d|Note\.|Proof\.|\Z))',
        ],
        # Proposition
        'proposition': [
            r'Proposition\s+([\d.]+)\s*\(([^)]+)\)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Proposition\s+\d|Note\.|Proof\.|\Z))',
            r'Proposition\s+([\d.]+)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Proposition\s+\d|Note\.|Proof\.|\Z))',
        ],
        # Corollary
        'corollary': [
            r'Corollary\s+([\d.]+)\s*\(([^)]+)\)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Corollary\s+\d|Note\.|Proof\.|\Z))',
            r'Corollary\s+([\d.]+)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Corollary\s+\d|Note\.|Proof\.|\Z))',
        ],
        # Proof patterns - ends with box symbol or next section
        'proof': [
            r'Proof\.\s*(.*?)(?:□|∎|\n\n(?=Definition\.|Theorem\s+\d|Lemma\s+\d|Note\.|Example))',
        ],
        # Note: "Note." followed by explanatory content
        'note': [
            r'Note\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Lemma\s+\d|Note\.|Proof\.|Example|\Z))',
        ],
        # Example patterns
        'example': [
            r'Example\s+([\d.]+)\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Example\s+\d|Note\.|Exercise|\Z))',
            r'Example\.\s*(.*?)(?=(?:Definition\.|Theorem\s+\d|Note\.|Exercise|\Z))',
        ],
        # Exercise patterns
        'exercise': [
            r'Exercise\s+([\d.]+)\.\s*(.*?)(?=(?:Exercise\s+\d|Solution|\Z))',
        ],
    }

    for item_type, regex_list in patterns.items():
        for pattern in regex_list:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                title = None
                content = None

                if isinstance(match, tuple):
                    if len(match) == 3:
                        # Pattern: (number, name, content) e.g., Theorem 2.1.3 (Name)
                        number, name, content = match
                        title = f"{name}" if name else f"{item_type.capitalize()} {number}"
                        content = content.strip()
                    elif len(match) == 2:
                        # Pattern: (number, content) e.g., Theorem 2.1.3.
                        number, content = match
                        title = f"{item_type.capitalize()} {number}"
                        content = content.strip()
                    else:
                        # Single group
                        content = match[-1].strip()
                else:
                    content = match.strip()

                if content and len(content) > 20:  # Skip very short matches
                    sections.append({
                        'type': item_type,
                        'title': title,
                        'content': clean_pdf_text(content)
                    })

    return sections


def clean_pdf_text(text: str) -> str:
    """Clean up PDF extracted text."""
    # Fix common PDF extraction issues
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)  # Fix hyphenation
    text = re.sub(r'\n(?!\n)', ' ', text)  # Join single newlines
    text = re.sub(r' +', ' ', text)  # Normalize spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Normalize paragraph breaks
    return text.strip()


def process_pdf_to_raw_chunks(pdf_path: Path, chunk_size: int = 2000) -> list[dict]:
    """
    Extract text from PDF and split into chunks for manual annotation.
    This is useful when automatic parsing doesn't work well.
    """
    text = extract_pdf_text(pdf_path)
    chunks = []

    # Split by paragraphs first, then combine into chunks
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) < chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append({
                    'raw_text': current_chunk.strip(),
                    'instruction': '',  # To be filled manually
                    'output': '',  # To be filled manually
                    'source': str(pdf_path.name)
                })
            current_chunk = para + "\n\n"

    if current_chunk.strip():
        chunks.append({
            'raw_text': current_chunk.strip(),
            'instruction': '',
            'output': '',
            'source': str(pdf_path.name)
        })

    return chunks


def process_pdf_file(pdf_path: Path) -> list[dict]:
    """Process a PDF file and extract training pairs."""
    print(f"  Extracting text from PDF...")
    text = extract_pdf_text(pdf_path)

    print(f"  Parsing sections...")
    sections = parse_pdf_sections(text)

    # Convert sections to training pairs
    all_pairs = []
    theorems_pending_proof = []

    for section in sections:
        item = ExtractedItem(
            item_type=section['type'],
            title=section['title'],
            content=section['content']
        )

        # Handle theorem-proof pairing
        if section['type'] in ('theorem', 'lemma', 'proposition', 'corollary'):
            theorems_pending_proof.append(item)
        elif section['type'] == 'proof' and theorems_pending_proof:
            # Attach proof to most recent theorem
            theorems_pending_proof[-1].associated_proof = section['content']
            pairs = generate_instruction_pairs(theorems_pending_proof.pop())
            for pair in pairs:
                pair['source_file'] = str(pdf_path.name)
            all_pairs.extend(pairs)
        else:
            # Process remaining pending theorems without proofs
            for thm in theorems_pending_proof:
                pairs = generate_instruction_pairs(thm)
                for pair in pairs:
                    pair['source_file'] = str(pdf_path.name)
                all_pairs.extend(pairs)
            theorems_pending_proof = []

            # Process current item
            pairs = generate_instruction_pairs(item)
            for pair in pairs:
                pair['source_file'] = str(pdf_path.name)
            all_pairs.extend(pairs)

    # Don't forget remaining theorems
    for thm in theorems_pending_proof:
        pairs = generate_instruction_pairs(thm)
        for pair in pairs:
            pair['source_file'] = str(pdf_path.name)
        all_pairs.extend(pairs)

    return all_pairs


def export_for_manual_annotation(pdf_path: Path, output_path: Path):
    """
    Export PDF content as chunks for manual annotation.
    Creates a JSON file that can be edited to create training pairs.
    """
    chunks = process_pdf_to_raw_chunks(pdf_path)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(chunks)} chunks to {output_path}")
    print("Edit this file to add 'instruction' and 'output' fields for each chunk.")
    print("Then run: python prepare_data.py --annotated_file <file> --output_file training_data.jsonl")


def process_annotated_file(filepath: Path) -> list[dict]:
    """Process a manually annotated JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    pairs = []
    for chunk in chunks:
        if chunk.get('instruction') and chunk.get('output'):
            pairs.append({
                'instruction': chunk['instruction'],
                'input': chunk.get('input', ''),
                'output': chunk['output'],
                'source_file': chunk.get('source', 'annotated')
            })

    return pairs


def main():
    parser = argparse.ArgumentParser(description='Convert course materials to training data')
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing LaTeX files')
    parser.add_argument('--output_file', type=str, default='training_data.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--problem_sets_dir', type=str, default=None,
                        help='Optional separate directory for problem sets')
    parser.add_argument('--pdf', type=str, default=None,
                        help='PDF file to process (textbook)')
    parser.add_argument('--export_for_annotation', action='store_true',
                        help='Export PDF chunks for manual annotation instead of auto-parsing')
    parser.add_argument('--annotated_file', type=str, default=None,
                        help='Process a manually annotated JSON file')
    args = parser.parse_args()

    all_pairs = []

    # Process PDF if specified
    if args.pdf:
        pdf_path = Path(args.pdf)
        if not pdf_path.exists():
            print(f"Error: PDF file not found: {pdf_path}")
            return

        if args.export_for_annotation:
            # Export for manual annotation
            annotation_file = Path(args.output_file).with_suffix('.json')
            export_for_manual_annotation(pdf_path, annotation_file)
            return
        else:
            # Auto-parse PDF
            print(f"Processing PDF: {pdf_path}...")
            pairs = process_pdf_file(pdf_path)
            all_pairs.extend(pairs)
            print(f"  Extracted {len(pairs)} training pairs")

    # Process annotated file if specified
    if args.annotated_file:
        annotated_path = Path(args.annotated_file)
        print(f"Processing annotated file: {annotated_path}...")
        pairs = process_annotated_file(annotated_path)
        all_pairs.extend(pairs)
        print(f"  Loaded {len(pairs)} training pairs")

    # Process LaTeX files if input_dir specified
    if args.input_dir:
        input_path = Path(args.input_dir)
        print(f"Processing LaTeX files from {input_path}...")
        for latex_file in input_path.glob('**/*.tex'):
            print(f"  Processing {latex_file.name}...")
            pairs = process_latex_file(latex_file)
            all_pairs.extend(pairs)
            print(f"    Extracted {len(pairs)} training pairs")

    # Process problem sets if specified
    if args.problem_sets_dir:
        ps_path = Path(args.problem_sets_dir)
        print(f"\nProcessing problem sets from {ps_path}...")
        for latex_file in ps_path.glob('**/*.tex'):
            print(f"  Processing {latex_file.name}...")
            pairs = process_problem_set(latex_file)
            all_pairs.extend(pairs)
            print(f"    Extracted {len(pairs)} training pairs")

    if not all_pairs:
        print("No training pairs extracted. Please specify --pdf, --input_dir, or --annotated_file")
        return

    # Write output
    output_path = Path(args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')

    print(f"\nTotal training pairs: {len(all_pairs)}")
    print(f"Output written to {output_path}")

    # Print summary by type
    print("\nSummary by source file:")
    from collections import Counter
    sources = Counter(p.get('source_file', 'unknown') for p in all_pairs)
    for source, count in sources.most_common():
        print(f"  {source}: {count} pairs")


if __name__ == '__main__':
    main()
