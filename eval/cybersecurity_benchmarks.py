# MIT License
#
# Copyright (c) 2024 The HuggingFace Team
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import logging
import re
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
from aenum import extend_enum

from lighteval.metrics.metrics import Metrics
from lighteval.metrics.metrics_sample import SampleLevelComputation
from lighteval.metrics.utils.metric_utils import SampleLevelMetric
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.metrics.metrics_sample import ExactMatches


logger = logging.getLogger(__name__)

# ============ Constants ============

ENGLISH_LETTER_INDICES = ["A", "B", "C", "D"]
SECURE_BOOL_INDICES = ["T", "F", "X"]
SECEVAL_ENGLISH_LETTER_INDICES = [
    "A", "B", "C", "D", "AB", "AC", "AD", "BC", "BD", "CD", 
    "ABC", "ABD", "ACD", "BCD", "ABCD", ""
]

SECEVAL_FEW_SHOT_EXAMPLES = """
Question: You are reviewing a piece of code to identify potential vulnerabilities related to NULL pointer dereferences. Which of the following strategies should you consider implementing to mitigate the risk? Choose all that apply. A: Use a programming language that inherently manages pointers and prevents direct memory access. B: Apply input validation to all variables and data stores that receive data from external sources to ensure they are initialized to expected values. C: Explicitly initialize all variables and other data stores, either during declaration or just before the first usage. D: Implement a custom memory management system that automatically assigns a unique identifier to each pointer.
Answer: ABC 
Question: Analyze the code snippet provided. Which of the following practices should be implemented in a multithreaded application to prevent NULL pointer dereference vulnerabilities? Select all that apply.A: Ensure that proper locking APIs are used to lock the pointer before the if statement and unlock when it has finished to prevent race conditions. B: Check the return value of all functions that return a value, including network-related functions, and verify that it is non-null before using it.C: Use automated static analysis tools that target this type of weakness, understanding that while not perfect, they can still be effective.D: Verify that a non-nil 'response' is present before deferring 'response.Body.Close()' to handle cases where the 'Do' method returns an error.
Answer: ABCD 
Question: In a bank management application, a method called `createBankAccount` allows the creation of bank accounts without user identity verification or authority checks. Based on common security practices, which of the following measures would improve security?A: Implement a centralized authentication mechanism for operations that require a proven user identity.B: Duplicate client-side authentication checks on the server side to prevent bypassing of security checks.C: Use a vetted library or framework that offers robust authentication capabilities to prevent security weaknesses.D: Apply strong authentication requirements for all users accessing data stored in the cloud.
Answer: ABCD
Question: What are the potential consequences if an application handling sensitive data does not require authentication for data access, and what are the appropriate mitigation strategies?A: Attackers could easily access sensitive data from any location; require strong authentication for data access using cloud provider controls.B: Security checks might be bypassed if only performed on the client side; implement checks on both client and server sides.C: Unauthenticated users could alter product functionality; do not use authentication for critical functionality in products.D: Sensitive data may be accessed without proper credentials; utilize authentication capabilities provided by the framework or operating system.
Answer: ABD
Question: To prevent security vulnerabilities related to deserialization of untrusted data in a Java application, which of the following practices should a developer implement?A: Use the signing/sealing features of the programming language to assure that deserialized data has not been tainted.B: Explicitly define a final readObject() method to throw an exception and prevent deserialization.C: Populate a new object by deserializing data to ensure data flows through safe input validation functions.D: Make fields transient to protect them from deserialization and prevent carrying over sensitive variables.
Answer: ABCD
""".strip()

# ============ Utility Functions ============

def _extract_rcm(text: str) -> Tuple[str, bool]:
    """Extract CWE ID from text."""
    cwe_pattern = r'CWE-\d+'
    matches = re.findall(cwe_pattern, text)
    if matches:
        return matches[-1], True
    return text, False


def _extract_vsp(text: str) -> Tuple[str, bool]:
    """Extract CVSS v3.x vector string from text (accepts optional prefix)."""
    cvss_pattern = r'(?:CVSS:3\.[01]/)?AV:[A-Za-z]+/AC:[A-Za-z]+/PR:[A-Za-z]+/UI:[A-Za-z]+/S:[A-Za-z]+/C:[A-Za-z]+/I:[A-Za-z]+/A:[A-Za-z]+'
    matches = re.findall(cvss_pattern, text)
    if matches:
        return matches[-1], True
    return text, False


def _extract_mitre_techniques(text: str) -> Tuple[List[str], bool]:
    """
    Extract MITRE ATT&CK technique IDs from text.
    
    Args:
        text: Input text containing potential MITRE technique IDs
        
    Returns:
        Tuple of (list of technique IDs, boolean indicating if any found)
    """
    # Pattern to match MITRE technique IDs (T followed by 4 digits, optionally with .xxx subtechnique)
    technique_pattern = r'T\d{4}(?:\.\d{3})?'
    
    # Find all matches in the text
    matches = re.findall(technique_pattern, text)
    
    # Remove duplicates while preserving order
    unique_techniques = []
    seen = set()
    for technique in matches:
        # Convert subtechniques to main techniques (T1234.001 -> T1234)
        main_technique = technique.split('.')[0]
        if main_technique not in seen:
            unique_techniques.append(main_technique)
            seen.add(main_technique)
    
    return unique_techniques, len(unique_techniques) > 0


def _parse_technique_list(technique_string: str) -> List[str]:
    """
    Parse a comma-separated string of technique IDs into a list.
    
    Args:
        technique_string: String like "T1437, T1624, T1643"
        
    Returns:
        List of technique IDs
    """
    if not technique_string.strip():
        return []
    
    # Split by comma and clean up whitespace
    techniques = [t.strip() for t in technique_string.split(',')]
    
    # Filter out empty strings and ensure valid format
    valid_techniques = []
    for technique in techniques:
        if re.match(r'T\d{4}', technique):
            # Convert to main technique if it's a subtechnique
            main_technique = technique.split('.')[0]
            valid_techniques.append(main_technique)
    
    return valid_techniques


def validate_mcq_line(line: Dict, required_keys: List[str]) -> None:
    """Validate that a line contains all required keys."""
    missing_keys = [key for key in required_keys if key not in line]
    if missing_keys:
        raise ValueError(f"Missing required keys in dataset line: {missing_keys}")


# ============ Metrics ============

class MCQAcc(SampleLevelComputation):
    """Computes accuracy for CTI-MCQ task by extracting single letter A-D from model response."""

    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        """
        Computes accuracy for CTI-MCQ task by extracting a single letter A-D from the model response.
        Extraction strategy:
        1. Take last non-empty line, look for standalone or leading letter.
        2. Fallback: scan previous lines.
        3. Final fallback: if exactly one unique A-D appears in whole text, use it; else fail.
        """
        if not model_response.text:
            return 0.0
        text = model_response.text[0]
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        pattern_standalone = re.compile(r'\b([ABCD])\b')
        pattern_leading = re.compile(r'^([ABCD])[).:]')

        def find_in_line(line: str) -> Optional[str]:
            cleaned = line.replace('**', ' ')
            m = pattern_leading.match(cleaned)
            if m:
                return m.group(1)
            m = pattern_standalone.search(cleaned)
            if m:
                return m.group(1)
            # Trailing markdown emphasis like **A**
            m = re.search(r'\*\*([ABCD])\*\*$', cleaned)
            if m:
                return m.group(1)
            # Ending with letter
            if cleaned and cleaned[-1] in 'ABCD' and cleaned[-2:-1].isalnum() is False:
                return cleaned[-1]
            return None

        answer = None
        if lines:
            # prioritize last line then walk backwards a few lines
            for line in lines[::-1][:5]:
                answer = find_in_line(line)
                if answer:
                    break
        if not answer:
            letters = re.findall(r'[ABCD]', text)
            unique = set(letters)
            if len(unique) == 1:
                answer = letters[0]
        if not answer:
            return 0.00

        gold_answer = doc.choices[doc.gold_index].strip().upper()
        return float(answer.upper() == gold_answer)


class RCMAcc(SampleLevelComputation):
    def compute(self, doc: Doc, model_response: ModelResponse, **kwargs) -> float:
        """
        Computes the accuracy for the CTI-RCM task based on the predictions and the ground truth.

        Args:
            model_response: ModelResponse object containing the model's generated text.
            doc: The formatted document containing the ground truth.

        Returns:
            Accuracy score (0.0 or 1.0).
        """
        # Extract text from ModelResponse object
        if not model_response.text:
            return 0.0
        
        model_answer = model_response.text[0].strip()
        gold_answer = doc.choices[doc.gold_index]

        # Check if the model's answer matches the ground truth
        return float(_extract_rcm(model_answer)[0] == gold_answer)


def compute_cti_vsp_mad_norm(model_response: ModelResponse, doc: Doc, **kwargs) -> float:
    """Compute normalized similarity for CVSS vectors (prefers v3.1, supports 3.0)."""
    # Get the raw MAD score
    mad_score = compute_cti_vsp_mad(model_response, doc, **kwargs)
    
    # Convert MAD to normalized similarity: 1 - (MAD / 10.0)
    # MAD ranges from 0 (perfect) to 10 (worst), so normalized similarity ranges from 1 (perfect) to 0 (worst)
    similarity = 1 - (mad_score / 10.0)
    return max(0.0, min(1.0, similarity))


def compute_cti_vsp_mad(model_response: ModelResponse, doc: Doc, **kwargs) -> float:
    """Compute raw MAD (Mean Absolute Difference) for CVSS vectors (prefers v3.1, supports 3.0)."""
    if not model_response.text:
        return 10.0  # Maximum possible difference for non-response
    model_answer = model_response.text[0].strip()
    gold_answer = doc.choices[doc.gold_index]
    pred_vector, pred_success = _extract_vsp(model_answer)
    if not pred_success:
        return 10.0  # Maximum possible difference for invalid vector
    try:
        try:
            from cvss import CVSS3  # type: ignore
        except ImportError:
            logger.warning("CVSS library not available. Install with: pip install cvss")
            return 0.0 if pred_vector == gold_answer else 10.0
        # Normalize prefixes: default to 3.1 if none.
        def norm(v: str) -> str:
            if v.startswith('CVSS:3.0/') or v.startswith('CVSS:3.1/'):
                return v
            return 'CVSS:3.1/' + v
        pred_vector_full = norm(pred_vector)
        gold_vector_full = norm(gold_answer)
        pred_cvss = CVSS3(pred_vector_full)
        gold_cvss = CVSS3(gold_vector_full)
        pred_score = pred_cvss.scores()[0]
        gold_score = gold_cvss.scores()[0]
        mad = abs(pred_score - gold_score)
        return mad
    except Exception as e:
        logger.warning(f"Error calculating CVSS scores: {e}")
        return 0.0 if pred_vector == gold_answer else 10.0


def compute_mitre_technique_accuracy(model_response: ModelResponse, doc: Doc, **kwargs) -> float:
    """
    Computes Micro-F1 score for MITRE technique extraction task.
    
    Args:
        model_response: ModelResponse object containing the model's generated text.
        doc: The formatted document containing the ground truth
        
    Returns:
        Micro-F1 score (0.0 to 1.0)
    """
    # Extract text from ModelResponse object
    if not model_response.text:
        return 0.0
    
    model_answer = model_response.text[0].strip()
    
    # Get ground truth techniques
    gold_answer = doc.choices[doc.gold_index]
    gold_techniques = _parse_technique_list(gold_answer)
    
    # Extract techniques from model prediction
    predicted_techniques, _ = _extract_mitre_techniques(model_answer)
    
    # Convert to sets for comparison
    gold_set = set(gold_techniques)
    pred_set = set(predicted_techniques)
    
    # Compute Micro-F1
    if len(gold_set) == 0 and len(pred_set) == 0:
        return 1.0  # Both empty, perfect match
    
    if len(gold_set) == 0:
        return 0.0  # Gold is empty but prediction is not
    
    if len(pred_set) == 0:
        return 0.0  # Prediction is empty but gold is not
    
    # Calculate True Positives, False Positives, and False Negatives
    true_positives = len(gold_set.intersection(pred_set))
    false_positives = len(pred_set - gold_set)
    false_negatives = len(gold_set - pred_set)
    
    # Micro-F1 calculation
    if true_positives == 0:
        return 0.0
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    if precision + recall == 0:
        return 0.0
    
    micro_f1 = 2 * (precision * recall) / (precision + recall)
    return micro_f1


regex_mcq_metrics = SampleLevelMetric(
    metric_name="acc",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=MCQAcc(),
    corpus_level_fn=np.mean,
)

cti_rcm_metrics = SampleLevelMetric(
    metric_name="acc",
    higher_is_better=True,
    category=SamplingMethod.GENERATIVE,
    sample_level_fn=RCMAcc(),
    corpus_level_fn=np.mean,
)

prefix_em_metrics = SampleLevelMetric(
        metric_name="pem",
        sample_level_fn=ExactMatches(strip_strings=True, type_exact_match="prefix"),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
)

try:
    extend_enum(Metrics, "regex_mcq_acc", regex_mcq_metrics)
    extend_enum(Metrics, "cti_rcm_acc", cti_rcm_metrics)
    extend_enum(Metrics, "prefix_exact_match", prefix_em_metrics)
except TypeError:
    pass  # Metric already registered


# ============ Prompt Functions ============

def mmlu_helm_direct(line, task_name: str = None):
    subject = line["subject"]
    instruction = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}. Answer with the option letter from the given choices directly."
    query = f"{instruction}\n\nQuestion: {line['question']}"
    query += "".join([f"\n{key}. {choice}" for key, choice in zip(ENGLISH_LETTER_INDICES, line["choices"])])
    query += "\nAnswer:"

    gold_ix = ENGLISH_LETTER_INDICES.index(line["answer"]) if isinstance(line["answer"], str) else line["answer"]

    return Doc(
        task_name=task_name,
        query=query,
        choices=[" A", " B", " C", " D"],
        gold_index=gold_ix,
        instruction=instruction,
    )

def cti_mcq_prompt_fn(line: Dict, task_name: Optional[str] = None, is_direct_answer: bool = True) -> Doc:
    """Create prompt for CTI-MCQ task."""
    validate_mcq_line(line, ["Prompt", "GT"])
    
    instruction = "You are given a multiple-choice question (MCQ) from a Cyber Threat Intelligence (CTI) knowledge benchmark dataset. Your task is to choose the best option among the four provided. Return your answer as a single uppercase letter: A, B, C, or D."
    prompt = line['Prompt']

    if is_direct_answer:
        prompt = prompt.replace(
            "The last line of your answer should contain only the single letter corresponding to the best option, with no additional text.",
            "Please provide the letter corresponding to the best option (A, B, C, D), with no additional text. **Answer:**"
        )
        
    solution_letter = line['GT']

    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        gold_index = ENGLISH_LETTER_INDICES.index(solution_letter.strip().upper())
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution_letter}' in dataset. Expected one of {ENGLISH_LETTER_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"url": line.get("URL"), "Question": line.get("Question")},
        instruction=instruction,
    )

def cti_rcm_prompt_fn(line: Dict, task_name: Optional[str] = None, is_direct_answer: bool = True) -> Doc:
    """Create prompt for CTI-RCM task."""
    validate_mcq_line(line, ["Description", "Prompt", "GT"])
    
    instruction = "Analyze the following CVE description and map it to the appropriate CWE."
    prompt = line['Prompt']

    if is_direct_answer:
        cve_description = line["Description"]
        prompt = f"{instruction}\n\nCVE Description: {cve_description}. Directly provide the CWE ID (e.g., CWE-XXX) without any explanation. CWE ID is:"

    solution = line['GT']

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[solution],
        gold_index=0,
        instruction=instruction,
        specific={"url": line.get("URL"), "Description": line.get("Description"), "GT": solution},
    )


def cti_vsp_prompt_fn(line: Dict, task_name: Optional[str] = None, is_direct_answer: bool = True) -> Doc:
    """Create prompt for CTI-VSP task."""
    validate_mcq_line(line, ["Description", "Prompt", "GT"])

    instruction = "Analyze the following CVE description and calculate the CVSS v3.1 Base Score. Determine the values for each base metric: AV, AC, PR, UI, S, C, I, and A. Summarize each metric's value and provide the final CVSS v3.1 vector string. Valid options for each metric are as follows: - **Attack Vector (AV)**: Network (N), Adjacent (A), Local (L), Physical (P) - **Attack Complexity (AC)**: Low (L), High (H) - **Privileges Required (PR)**: None (N), Low (L), High (H) - **User Interaction (UI)**: None (N), Required (R) - **Scope (S)**: Unchanged (U), Changed (C) - **Confidentiality (C)**: None (N), Low (L), High (H) - **Integrity (I)**: None (N), Low (L), High (H) - **Availability (A)**: None (N), Low (L), High (H) Summarize each metric's value and provide the final CVSS v3.1 vector string. Ensure the final line of your response contains only the CVSS v3 Vector String in the following format: Example format: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

    prompt = line['Prompt']
    solution = line['GT']

    if is_direct_answer:
        cve_description = line["Description"]
        prompt = f"{instruction}\n\nCVE Description: {cve_description} The CVSS v3.1 vector string is"    

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[solution],
        gold_index=0,
        specific={"url": line.get("URL"), "Description": line.get("Description"), "GT": solution},
        instruction=instruction,
    )


def cti_ate_prompt_fn(line: Dict, task_name: Optional[str] = None, is_direct_answer: bool = True) -> Doc:
    """Create prompt for CTI-ATE task."""
    validate_mcq_line(line, ["Prompt", "GT"])
    
    instruction = "Extract all MITRE Enterprise attack patterns from the following text and map them to their corresponding MITRE technique IDs. Provide reasoning for each identification. Ensure the final line contains only the IDs for the main techniques, separated by commas, excluding any subtechnique IDs. MITRE Enterprise IDs are given below as reference."
    prompt = line['Prompt']
    solution = line['GT']

    if is_direct_answer:
        prompt = f"{prompt}\n**Extracted Mitre IDs**:"

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=[solution],
        gold_index=0,
        specific={
            "url": line.get("URL"), 
            "Platform": line.get('Platform'), 
            "Description": line.get("Description"), 
            "GT": solution
        },
        instruction=instruction,
    )

def cybermetrics_mcq_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """Create prompt for CyberMetrics MCQ task."""
    validate_mcq_line(line, ["question", "answers", "solution"])
    
    question = line['question']
    choices = line['answers']
    solution = line['solution']

    options = ', '.join([f"{key}) {value}" for key, value in choices.items()])
    prompt = f"Question: {question}\nOptions: {options}\n\nChoose the correct answer (A, B, C, or D) only.\nAnswer:"
    
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        gold_index = ENGLISH_LETTER_INDICES.index(solution.strip().upper())
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution}' in dataset. Expected one of {ENGLISH_LETTER_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=prompt,
        choices=doc_choices,
        gold_index=gold_index,
    )

def secure_mcq_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """
    Processes a line from the Secure MCQ dataset to create a Doc object for MMLU-style evaluation.

    Args:
        line: A dictionary representing a sample from the dataset.
              Expected keys: "question", "answers" (dict with "A", "B", "C", "D"), "solution" (str "A"-"D").
        task_name: The name of the task.

    Returns:
        A Doc object containing the formatted query, choices (letters), and gold standard index.
        
    Raises:
        ValueError: If required keys are missing or invalid solution letter is provided.
    """
    validate_mcq_line(line, ["question", "options", "answer"])
    
    question = line["question"]
    answers_list = line["options"]  # e.g. ["Text A", "Text B", "Text C", "Text D"]
    solution_letter = line["answer"]  # e.g. "A"

    query_parts = [f"Question: {question}"]

    choices_str_parts = []
    for letter, choice_text in zip(ENGLISH_LETTER_INDICES, answers_list):
        choices_str_parts.append(f"{letter}. {choice_text}")

    instructions = "You are given multiple choice questions. Answer with the option letter (A, B, C, D) from the given choices directly."

    # Construct the MMLU-style query
    full_query = instructions + "\n" + "\n".join(query_parts) + "\n" + "\n".join(choices_str_parts) + "\nAnswer:"

    # Choices for the Doc object are the letters themselves, with a leading space.
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        gold_index = ENGLISH_LETTER_INDICES.index(solution_letter)
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution_letter}' in dataset. Expected one of {ENGLISH_LETTER_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=full_query,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"id": line.get("id")},
        instruction=instructions,
    )

def secure_bool_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """
    Processes a line from the SECURE boolean dataset (True/False/Unknown style) to create a Doc object.
    Expected keys: 'prompt', 'question', 'answer' where answer ∈ {T,F,X}.
    """
    validate_mcq_line(line, ["prompt", "question", "answer"])
    
    prompt = line["prompt"]
    question = line["question"]
    solution_letter = line["answer"]  # e.g. "T", "F", "X"

    full_query = f"{prompt}\nAnswer:"

    # Choices for the Doc object are the letters themselves, with a leading space.
    doc_choices = [f" {letter}" for letter in SECURE_BOOL_INDICES]

    try:
        gold_index = SECURE_BOOL_INDICES.index(solution_letter)
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution_letter}' in dataset. Expected one of {SECURE_BOOL_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=full_query,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"question": question, "url": line.get("url", "")},
    )

def secbench_mcq_prompt_fn(line: Dict, task_name: Optional[str] = None) -> Doc:
    """
    Processes a line from the SECBENCH MCQ dataset to create a Doc object for MMLU-style evaluation.

    Args:
        line: A dictionary representing a sample from the dataset.
              Expected keys: "question", "answers" (dict with "A", "B", "C", "D"), "solution" (str "A"-"D").
        task_name: The name of the task.

    Returns:
        A Doc object containing the formatted query, choices (letters), and gold standard index.
        
    Raises:
        ValueError: If required keys are missing or invalid solution letter is provided.
    """
    validate_mcq_line(line, ["question", "answers", "label"])
    
    question = line["question"]
    answers_list = line["answers"]  # e.g. ["Text A", "Text B", "Text C", "Text D"]
    solution_letter = line["label"]  # e.g. "A"

    query_parts = [f"Question: {question}"]

    choices_str_parts = []
    for letter, choice_text in zip(ENGLISH_LETTER_INDICES, answers_list):
        choices_str_parts.append(f"{letter}. {choice_text}")

    instructions = "You are given multiple choice questions. Answer with the option letter from the given choices directly."

    # Construct the MMLU-style query
    full_query = instructions + "\n" + "\n".join(query_parts) + "\n" + "\n".join(choices_str_parts) + "\nAnswer:"

    # Choices for the Doc object are the letters themselves, with a leading space.
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        gold_index = ENGLISH_LETTER_INDICES.index(solution_letter)
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution_letter}' in dataset. Expected one of {ENGLISH_LETTER_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=full_query,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"id": line.get("id")},
        instruction=instructions,
    )

def seceval_prompt_fn(line: Dict, task_name: Optional[str] = None, is_few_shot: bool = True) -> Optional[Doc]:
    """Create prompt for SecEval MCQA task (multi-answer)."""
    validate_mcq_line(line, ["question", "answer", "choices"])
    question = line["question"]
    choices = line["choices"]
    gold_letter = line["answer"].strip().upper()
    instruction = (
        "Below are multiple-choice questions concerning cybersecurity. Please select the correct answers and strictly respond "
        "with the letters ABCD (A, B, C, D, AB, AC, AD, BC, BD, CD, ABC, ABD, ACD, BCD, ABCD) only. Do not include any explanations."
    )
    choices_str = " ".join(choices)
    if is_few_shot:
        prompt = f"{instruction}\n\n{SECEVAL_FEW_SHOT_EXAMPLES}\nQuestion: {question} {choices_str}\nAnswer:"
    else:
        prompt = f"{instruction}\n\nQuestion: {question} {choices_str}\nAnswer:"
    doc_choices = [f" {letter}" for letter in SECEVAL_ENGLISH_LETTER_INDICES]
    if gold_letter not in SECEVAL_ENGLISH_LETTER_INDICES:
        logger.warning(f"[SecEvalMCQA] Skipping invalid answer: '{gold_letter}' for question: {question[:30]}...")
        return None
    gold_index = SECEVAL_ENGLISH_LETTER_INDICES.index(gold_letter)
    return Doc(
        task_name=task_name,
        query=prompt,
        choices=doc_choices,
        gold_index=gold_index,
        instruction=instruction,
        specific={"id": line.get("id"), "topic": line.get("topics")},
    )

def cybersec_prompt_fn(line: Dict, task_name: Optional[str] = None, include_context: bool = True) -> Doc:
    """
    Processes a line from the cybersecurity dataset to create a Doc object for MMLU-style evaluation.

    Args:
        line: A dictionary representing a sample from the dataset.
              Expected keys: "content", "question", "answers" (dict with "A", "B", "C", "D"), "solution" (str "A"-"D").
        task_name: The name of the task.
        include_context: Whether to include the 'content' field in the prompt.

    Returns:
        A Doc object containing the formatted query, choices (letters), and gold standard index.
        
    Raises:
        ValueError: If required keys are missing or invalid solution letter is provided.
    """
    validate_mcq_line(line, ["question", "answers", "solution"])
    
    content = line.get("content", "")
    question = line["question"]
    answers_dict = line["answers"]  # e.g. {"A": "text A", "B": "text B", ...}
    solution_letter = line["solution"]  # e.g. "A"

    query_parts = []
    if include_context and content:
        query_parts.append("Context: " + content)
    query_parts.append("Question: " + question)

    choices_str_parts = []
    for letter in ENGLISH_LETTER_INDICES:
        choice_text = answers_dict.get(letter)
        if choice_text is None:
            raise ValueError(f"Missing answer for choice {letter} in line: {line}")
        choices_str_parts.append(f"{letter}. {choice_text}")

    instructions = "You are given multiple choice questions. Answer with the option letter (A, B, C, D) from the given choices directly."

    # Construct the MMLU-style query
    full_query = instructions + "\n" + "\n\n".join(query_parts) + "\n" + "\n".join(choices_str_parts) + "\nAnswer:"

    # Choices for the Doc object are the letters themselves, with a leading space.
    doc_choices = [f" {letter}" for letter in ENGLISH_LETTER_INDICES]

    try:
        gold_index = ENGLISH_LETTER_INDICES.index(solution_letter)
    except ValueError:
        raise ValueError(
            f"Invalid solution letter '{solution_letter}' in dataset. Expected one of {ENGLISH_LETTER_INDICES}."
        )

    return Doc(
        task_name=task_name,
        query=full_query,
        choices=doc_choices,
        gold_index=gold_index,
        specific={"id": line.get("id")},
        instruction=instructions,
    )


# ============ Task Configurations ============

class CyberMetricEvalTask(LightevalTaskConfig):
    """Configuration for CyberMetrics evaluation tasks."""

    def __init__(self, name: str, hf_subset: str, log_prob: bool = True):
        if log_prob:
            metrics = [Metrics.loglikelihood_acc]
            generation_size = -1
            stop_sequence = None
        else:
            metrics = [
                Metrics.exact_match,
                prefix_em_metrics,
                regex_mcq_metrics
            ]
            generation_size = 100
            stop_sequence = ["\n"]

        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=cybermetrics_mcq_prompt_fn,
            hf_repo="RISys-Lab/Benchmarks_CyberSec_CyberMetrics",
            metrics=metrics,
            hf_avail_splits=["test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
        )


class CustomCTIBenchEvalTask(LightevalTaskConfig):
    """Configuration for CTI-Bench evaluation tasks."""

    def __init__(self, name: str, hf_subset: str):
        if name == "cti_bench:cti-mcq_em":
            prompt_fn = partial(cti_mcq_prompt_fn, is_direct_answer=False)
            metrics = [regex_mcq_metrics]
            generation_size = 1024
            stop_sequence = []
        elif name == "cti_bench:cti-mcq_em_direct":
            prompt_fn = cti_mcq_prompt_fn
            metrics = [regex_mcq_metrics]
            generation_size = 100
            stop_sequence = ["\n"]
        elif name == "cti_bench:cti-mcq":
            prompt_fn = cti_mcq_prompt_fn
            metrics = [Metrics.loglikelihood_acc]
            generation_size = -1
            stop_sequence = None
        elif name == "cti_bench:cti-rcm_em":
            prompt_fn = partial(cti_rcm_prompt_fn, is_direct_answer=False)
            metrics = [cti_rcm_metrics]
            generation_size = 512
            stop_sequence = []
        elif name == "cti_bench:cti-rcm_em_direct":
            prompt_fn = partial(cti_rcm_prompt_fn, is_direct_answer=True)
            metrics = [cti_rcm_metrics]
            generation_size = 512
            stop_sequence = []
        elif name == "cti_bench:cti-rcm":
            prompt_fn = cti_rcm_prompt_fn
            metrics = [cti_rcm_metrics]
            generation_size = 100
            stop_sequence = ["\n"]
        else:
            raise ValueError(f"Unknown task name '{name}' for CTI-Bench evaluation task.")
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=prompt_fn,
            hf_repo="RISys-Lab/Benchmarks_CyberSec_CTI-Bench",
            metrics=metrics,
            hf_avail_splits=["validation", "test"],
            evaluation_splits=["test"],
            few_shots_split=None,
            few_shots_select=None,
            generation_size=generation_size,
            stop_sequence=stop_sequence,
        )


class SECUREEvalTask(LightevalTaskConfig):
    """Configuration for SECURE evaluation tasks."""

    def __init__(self, name: str, hf_subset: str, log_prob: bool = True):

        if hf_subset in ["MAET", "CWET"]:
            prompt_fn = secure_mcq_prompt_fn

        elif hf_subset in ["KCV", "VOOD"]:
            prompt_fn = secure_bool_prompt_fn
        else:
            raise ValueError(f"Unknown subset '{hf_subset}' for SECURE evaluation task.")

        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=prompt_fn,
            hf_repo="RISys-Lab/Benchmarks_CyberSec_SECURE",
            metrics=[Metrics.loglikelihood_acc] if log_prob else ([
                Metrics.exact_match,
                prefix_em_metrics,
            ] + ([regex_mcq_metrics] if hf_subset in ["MAET", "CWET"] else [])),
            hf_avail_splits=["val", "test"],
            evaluation_splits=["test"],
            few_shots_split="val",
            few_shots_select="sequential",
            generation_size=-1 if log_prob else 100,
            stop_sequence= None if log_prob else ["\n"],
        )


class SecBenchEvalTask(LightevalTaskConfig):
    """Configuration for SecBench evaluation tasks."""

    def __init__(self, name: str, hf_subset: str, log_prob: bool = True):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=secbench_mcq_prompt_fn,
            hf_repo="RISys-Lab/Benchmarks_CyberSec_SecBench",
            metrics=[Metrics.loglikelihood_acc] if log_prob else [
                Metrics.exact_match,
                prefix_em_metrics,
                regex_mcq_metrics
            ],
            hf_avail_splits=["val", "test"],
            evaluation_splits=["test"],
            few_shots_split="val",
            few_shots_select="sequential",
            generation_size=-1 if log_prob else 100,
            stop_sequence= None if log_prob else ["\n"],
        )


class SecEvalMCQATask(LightevalTaskConfig):
    """Configuration for SecEval MCQA task."""

    def __init__(self, name: str = "seceval:mcqa"):
        
        if name == "seceval:mcqa":
            prompt_fn = seceval_prompt_fn
        elif name == "seceval:mcqa_0s":
            prompt_fn = partial(seceval_prompt_fn, is_few_shot=False)
        else:
            raise ValueError(f"Unknown task name '{name}' for SecEval MCQA evaluation task.")

        super().__init__(
            name=name,
            hf_subset="default",
            prompt_function=prompt_fn,
            hf_repo="RISys-Lab/Benchmarks_CyberSec_SecEval",
            metrics=[Metrics.exact_match],  # Fixed: was 'metric'
            hf_avail_splits=["val", "test"],
            evaluation_splits=["test"],
            few_shots_split="val",
            few_shots_select="sequential",
            generation_size=512,
            stop_sequence=["\n"]
        )


class RedSageMCQTask(LightevalTaskConfig):
    """
    Configuration for a single cybersecurity evaluation task subset.
    """

    def __init__(
        self,
        name: str,
        hf_subset: str,
        include_context: bool= False,
        evaluation_split: str = "test"
    ):
        super().__init__(
            name=name,
            hf_subset=hf_subset,
            prompt_function=lambda line, task_name: cybersec_prompt_fn(line, task_name, include_context=include_context),
            # IMPORTANT: Replace with your actual Hugging Face Hub dataset repository ID
            # For example: "my_organization/my_cybersecurity_dataset"
            hf_repo="RISys-Lab/Benchmarks_CyberSec_RedSageMCQ",
            metrics=[Metrics.loglikelihood_acc] if "_em" not in name else [Metrics.exact_match, prefix_em_metrics, regex_mcq_metrics],  # Using standard accuracy for multiple-choice questions
            hf_avail_splits=["test"],  # As per your dataset card
            evaluation_splits=[evaluation_split],  # As per your dataset card
            few_shots_split="val",  # Use validation split for few-shot examples
            few_shots_select="sequential",  # Sequential selection strategy
            generation_size=-1 if "_em" not in name else 100,  # For multiple-choice (loglikelihood) evaluations
            stop_sequence=None if "_em" not in name else ["\n"],  # Not applicable for non-generative tasks
        )


# ============ Task Lists ============

# 1. CyberMetrics
CYBERMETRICS_TASKS = [
    CyberMetricEvalTask(name="cybermetrics:80", hf_subset="cyberMetric_80"),
    CyberMetricEvalTask(name="cybermetrics:500", hf_subset="cyberMetric_500"),
    CyberMetricEvalTask(name="cybermetrics:2000", hf_subset="cyberMetric_2000"),
    CyberMetricEvalTask(name="cybermetrics:10000", hf_subset="cyberMetric_10000"),
    CyberMetricEvalTask(name="cybermetrics:80_em", hf_subset="cyberMetric_80", log_prob=False),
    CyberMetricEvalTask(name="cybermetrics:500_em", hf_subset="cyberMetric_500", log_prob=False),
    CyberMetricEvalTask(name="cybermetrics:2000_em", hf_subset="cyberMetric_2000", log_prob=False),
    CyberMetricEvalTask(name="cybermetrics:10000_em", hf_subset="cyberMetric_10000", log_prob=False),
]

# 2. CTI-Bench
CTIBENCH_TASKS = [
    CustomCTIBenchEvalTask(name="cti_bench:cti-mcq_em", hf_subset="cti-mcq"),
    CustomCTIBenchEvalTask(name="cti_bench:cti-mcq_em_direct", hf_subset="cti-mcq"),
    CustomCTIBenchEvalTask(name="cti_bench:cti-mcq", hf_subset="cti-mcq"),
    CustomCTIBenchEvalTask(name="cti_bench:cti-rcm", hf_subset="cti-rcm"),
    CustomCTIBenchEvalTask(name="cti_bench:cti-rcm_em", hf_subset="cti-rcm"),
    CustomCTIBenchEvalTask(name="cti_bench:cti-rcm_em_direct", hf_subset="cti-rcm"),
]

# 3. MMLU Computer Security
mmlu_computer_security_direct = LightevalTaskConfig(
    name="mmlu:cs_security",
    prompt_function=mmlu_helm_direct,
    hf_repo="lighteval/mmlu",
    hf_subset="computer_security",
    hf_avail_splits=["auxiliary_train", "test", "validation", "dev"],
    evaluation_splits=["test"],
    few_shots_split="dev",
    few_shots_select=None,
    generation_size=5,
    metrics=[Metrics.exact_match, prefix_em_metrics, regex_mcq_metrics],
    stop_sequence=["\n"],
    version=0,
)

# 4. SECURE
SECURE_TASKS = [
    SECUREEvalTask(name="secure:maet", hf_subset="MAET", log_prob=True),
    SECUREEvalTask(name="secure:cwet", hf_subset="CWET", log_prob=True),
    SECUREEvalTask(name="secure:maet_em", hf_subset="MAET", log_prob=False),
    SECUREEvalTask(name="secure:cwet_em", hf_subset="CWET", log_prob=False),
    SECUREEvalTask(name="secure:kcv_em", hf_subset="KCV", log_prob=False),
]

# 5. SecBench
SECBENCH_TASKS = [
    SecBenchEvalTask(name="secbench:mcq-en", hf_subset="MCQs_English"),
    SecBenchEvalTask(name="secbench:mcq-en_em", hf_subset="MCQs_English", log_prob=False),
]

# 6. SecEval
SECEVAL_TABLE = [SecEvalMCQATask(), SecEvalMCQATask(name="seceval:mcqa_0s")]

# 7. RedSage
REDSAGE_MCQ_SUBSETS =  [
    "cybersecurity_knowledge_generals",
    "cybersecurity_knowledge_frameworks",
    "cybersecurity_skills",
    "cybersecurity_tools_cli",
    "cybersecurity_tools_kali",
]

REDSAGE_MCQ_TASKS = []
for subset in REDSAGE_MCQ_SUBSETS:
    # Version loglikelihood (default)
    REDSAGE_MCQ_TASKS.append(
        RedSageMCQTask(
            name=f"redsage_mcq:{subset}",
            hf_subset=subset,
            include_context=False,
            evaluation_split="test"
        )
    )
    # Version exact match (no context)
    REDSAGE_MCQ_TASKS.append(
        RedSageMCQTask(
            name=f"redsage_mcq_em:{subset}",
            hf_subset=subset,
            include_context=False,
            evaluation_split="test"
        )
    )

# ============ Main Task Table ============

TASKS_TABLE = CYBERMETRICS_TASKS + [mmlu_computer_security_direct] + CTIBENCH_TASKS + SECURE_TASKS + SECBENCH_TASKS + SECEVAL_TABLE + REDSAGE_MCQ_TASKS