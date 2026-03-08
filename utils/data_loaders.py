"""
Data Loading Utilities for BERT Quantization Fairness Analysis
================================================================

This module provides functions to load and prepare all three datasets:
1. Jigsaw Toxic Comment Classification (Binary toxicity)
2. Bias in Bios (28 occupation classes)
3. Equity Evaluation Corpus (Counterfactual pairs)
"""

import pandas as pd
import numpy as np
from datasets import load_from_disk
from typing import Dict, List, Tuple
import os


# ============================================================================
# JIGSAW DATASET
# ============================================================================

def load_jigsaw_data(data_dir: str = 'data/jigsaw_data', 
                     sample_size: int = None) -> Tuple:
    """
    Load Jigsaw Toxic Comment Classification dataset.
    
    This is a multi-label dataset, but we convert it to binary:
    - Toxic (1): If ANY of the 6 toxic categories is 1
    - Non-Toxic (0): If ALL categories are 0
    
    Note: This dataset does NOT have identity columns in train.csv.
    For unbiased toxicity with demographics, you'd need:
    "jigsaw-unintended-bias-in-toxicity-classification"
    
    Args:
        data_dir: Path to jigsaw_data folder
        sample_size: Optional - limit number of samples for faster testing
        
    Returns:
        train_texts: List of comment texts
        train_labels: Binary labels (0=non-toxic, 1=toxic)
        test_texts: List of test comment texts
        test_labels: Binary test labels
    """
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    test_labels_path = os.path.join(data_dir, 'test_labels.csv')
    
    print(f"Loading Jigsaw data from {data_dir}...")
    
    # Load training data
    train_df = pd.read_csv(train_path)
    
    # Convert multi-label to binary
    # toxic + severe_toxic + obscene + threat + insult + identity_hate
    toxic_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    train_df['is_toxic'] = (train_df[toxic_columns].sum(axis=1) > 0).astype(int)
    
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    
    train_texts = train_df['comment_text'].astype(str).tolist()
    train_labels = train_df['is_toxic'].tolist()
    
    # Load test data
    test_df = pd.read_csv(test_path)
    test_labels_df = pd.read_csv(test_labels_path)
    
    # Merge test texts with labels
    test_df = test_df.merge(test_labels_df, on='id')
    
    # Remove samples with -1 labels (unlabeled)
    test_df = test_df[test_df['toxic'] != -1]
    
    # Convert to binary
    test_df['is_toxic'] = (test_df[toxic_columns].sum(axis=1) > 0).astype(int)
    
    if sample_size:
        test_df = test_df.sample(n=min(sample_size // 5, len(test_df)), random_state=42)
    
    test_texts = test_df['comment_text'].astype(str).tolist()
    test_labels = test_df['is_toxic'].tolist()
    
    print(f"Jigsaw loaded: {len(train_texts)} train, {len(test_texts)} test")
    print(f"Train toxic rate: {np.mean(train_labels):.2%}")
    print(f"Test toxic rate: {np.mean(test_labels):.2%}")
    
    # Note: This dataset doesn't have demographic info
    # For fairness analysis, we'd need the "unintended bias" version
    # For now, return dummy identity data
    train_identities = [{'race': 'unknown', 'gender': 'unknown'} for _ in train_texts]
    test_identities = [{'race': 'unknown', 'gender': 'unknown'} for _ in test_texts]
    
    return train_texts, train_labels, train_identities, test_texts, test_labels, test_identities


def load_jigsaw_unintended_bias(data_dir: str = 'data/jigsaw_bias_data',
                                sample_size: int = None) -> Tuple:
    """
    Load Jigsaw Unintended Bias in Toxicity Classification (preferred version).
    
    This version includes identity annotations for fairness analysis.
    
    Available identity columns:
    - male, female, transgender, other_gender
    - heterosexual, homosexual_gay_or_lesbian, bisexual, other_sexual_orientation
    - christian, jewish, muslim, hindu, buddhist, atheist, other_religion
    - black, white, asian, latino, other_race_or_ethnicity
    
    Note: This requires downloading the separate competition dataset:
    kagglehub.competition_download("jigsaw-unintended-bias-in-toxicity-classification")
    
    Args:
        data_dir: Path to the unintended bias dataset folder
        sample_size: Optional sample size limit
        
    Returns:
        Texts, binary labels, and identity annotations
    """
    train_path = os.path.join(data_dir, 'train.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Jigsaw Unintended Bias dataset not found at {train_path}\n"
            "Download it using: kagglehub.competition_download("
            "'jigsaw-unintended-bias-in-toxicity-classification')"
        )
    
    print(f"Loading Jigsaw Unintended Bias from {data_dir}...")
    
    df = pd.read_csv(train_path)
    
    # Binary toxicity label (>= 0.5 threshold)
    df['is_toxic'] = (df['target'] >= 0.5).astype(int)
    
    # Identity columns (values are 0 to 1, we threshold at 0.5)
    identity_cols = [
        'male', 'female', 'transgender', 'other_gender',
        'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation',
        'christian', 'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion',
        'black', 'white', 'asian', 'latino', 'other_race_or_ethnicity'
    ]
    
    # Fill NaN with 0
    for col in identity_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    texts = df['comment_text'].astype(str).tolist()
    labels = df['is_toxic'].tolist()
    
    # Extract identity annotations
    identities = []
    for _, row in df.iterrows():
        identity_dict = {col: int(row[col] >= 0.5) for col in identity_cols if col in df.columns}
        identities.append(identity_dict)
    
    print(f"Jigsaw Unintended Bias loaded: {len(texts)} samples")
    print(f"Toxic rate: {np.mean(labels):.2%}")
    
    return texts, labels, identities


# ============================================================================
# BIAS IN BIOS DATASET
# ============================================================================

def load_bias_in_bios(data_dir: str = 'data/bias_in_bios') -> Tuple:
    """
    Load Bias in Bios professional occupation dataset.
    
    Dataset structure:
    - 28 occupation classes (professor, surgeon, nurse, etc.)
    - Gender inferred from pronouns and names
    - Professional biographies as text
    
    Args:
        data_dir: Path to bias_in_bios folder
        
    Returns:
        train_texts: List of biography texts
        train_labels: Occupation labels (0-27)
        train_genders: Gender labels (0=female, 1=male, 2=other)
        test_texts, test_labels, test_genders: Same for test set
    """
    print(f"Loading Bias in Bios from {data_dir}...")
    
    # Load the dataset saved by HuggingFace datasets
    dataset = load_from_disk(data_dir)
    
    # Define occupation to label mapping (28 classes)
    occupations = [
        'accountant', 'architect', 'attorney', 'chiropractor', 'comedian',
        'composer', 'dentist', 'dietitian', 'dj', 'filmmaker',
        'interior_designer', 'journalist', 'model', 'nurse', 'painter',
        'paralegal', 'pastor', 'personal_trainer', 'photographer', 'physician',
        'poet', 'professor', 'psychologist', 'rapper', 'software_engineer',
        'surgeon', 'teacher', 'yoga_teacher'
    ]
    
    num_occupations = len(occupations)
    
    # Gender mapping — handles both string ('f'/'m') and int (0/1) formats
    gender_map = {'f': 0, 'F': 0, 'm': 1, 'M': 1, 0: 0, 1: 1}
    
    def process_split(split_data):
        """Process a dataset split (train/test/dev)."""
        texts = []
        labels = []
        genders = []
        
        for item in split_data:
            # Get biography text
            text = item.get('hard_text', item.get('bio', ''))
            if not text:
                continue
            
            # Get occupation label — already an int in LabHC/bias_in_bios
            profession = item.get('profession', item.get('title', None))
            if profession is None:
                continue
            if isinstance(profession, int):
                if profession < 0 or profession >= num_occupations:
                    continue
                label = profession
            else:
                # String label fallback
                occ_map = {occ: idx for idx, occ in enumerate(occupations)}
                if profession not in occ_map:
                    continue
                label = occ_map[profession]
            
            # Get gender
            raw_gender = item.get('gender', 2)
            gender = gender_map.get(raw_gender, 2)  # 2 for other/unknown
            
            texts.append(str(text))
            labels.append(label)
            genders.append(gender)
        
        return texts, labels, genders
    
    # Process train and test splits
    train_texts, train_labels, train_genders = process_split(dataset['train'])
    test_texts, test_labels, test_genders = process_split(dataset['test'])
    
    print(f"Bias in Bios loaded:")
    print(f"  Train: {len(train_texts)} samples")
    print(f"  Test: {len(test_texts)} samples")
    print(f"  Occupations: {len(occupations)}")
    print(f"  Gender distribution (train): Female={train_genders.count(0)}, "
          f"Male={train_genders.count(1)}, Other={train_genders.count(2)}")
    
    return train_texts, train_labels, train_genders, test_texts, test_labels, test_genders, occupations


# ============================================================================
# EQUITY EVALUATION CORPUS
# ============================================================================

def load_equity_evaluation_corpus(data_path: str = 'data/Equity-Evaluation-Corpus.csv') -> List[Tuple[str, str]]:
    """
    Load Equity Evaluation Corpus for counterfactual fairness testing.
    
    This corpus contains sentence pairs where only demographic terms differ:
    - "Alonzo feels angry" vs "Adam feels angry"
    - "She is a doctor" vs "He is a doctor"
    
    Used to compute Counterfactual Flip Rate (CFR).
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        List of (sentence1, sentence2) pairs
    """
    print(f"Loading Equity Evaluation Corpus from {data_path}...")
    
    df = pd.read_csv(data_path)
    
    # The corpus has templates like "<person subject> feels <emotion word>"
    # We need to create pairs by matching sentences with same template but different people
    
    pairs = []
    
    # Group by Template and Emotion word
    grouped = df.groupby(['Template', 'Emotion word'])
    
    for (template, emotion), group in grouped:
        sentences = group['Sentence'].tolist()
        
        # Create pairs from different demographic groups
        # For example: all African-American male names vs European male names
        if len(sentences) >= 2:
            # Simple pairing: just take consecutive pairs
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    pairs.append((sentences[i], sentences[i + 1]))
    
    print(f"Equity Evaluation Corpus loaded: {len(pairs)} counterfactual pairs")
    
    return pairs


def load_equity_corpus_advanced(data_path: str = 'data/Equity-Evaluation-Corpus.csv') -> Dict:
    """
    Advanced loader that creates structured counterfactual pairs.
    
    Returns pairs grouped by:
    - Race swaps (African-American ↔ European)
    - Gender swaps (male ↔ female)
    - Emotion categories
    
    Returns:
        Dictionary with keys: 'race_pairs', 'gender_pairs', 'all_pairs'
    """
    df = pd.read_csv(data_path)
    
    result = {
        'race_pairs': [],
        'gender_pairs': [],
        'all_pairs': []
    }
    
    # Group by template and emotion to find matching sentences
    for (template, emotion), group in df.groupby(['Template', 'Emotion word']):
        sentences_by_demo = {}
        
        for _, row in group.iterrows():
            key = (row['Race'], row['Gender'])
            sentences_by_demo[key] = row['Sentence']
        
        # Race swaps (same gender, different race)
        aa_male = sentences_by_demo.get(('African-American', 'male'))
        eur_male = sentences_by_demo.get(('European', 'male'))
        if aa_male and eur_male:
            result['race_pairs'].append((aa_male, eur_male))
            result['all_pairs'].append((aa_male, eur_male))
        
        aa_female = sentences_by_demo.get(('African-American', 'female'))
        eur_female = sentences_by_demo.get(('European', 'female'))
        if aa_female and eur_female:
            result['race_pairs'].append((aa_female, eur_female))
            result['all_pairs'].append((aa_female, eur_female))
        
        # Gender swaps (same race, different gender)
        if aa_male and aa_female:
            result['gender_pairs'].append((aa_male, aa_female))
            result['all_pairs'].append((aa_male, aa_female))
        
        if eur_male and eur_female:
            result['gender_pairs'].append((eur_male, eur_female))
            result['all_pairs'].append((eur_male, eur_female))
    
    print(f"Equity Corpus Advanced:")
    print(f"  Race pairs: {len(result['race_pairs'])}")
    print(f"  Gender pairs: {len(result['gender_pairs'])}")
    print(f"  Total pairs: {len(result['all_pairs'])}")
    
    return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_class_distribution(labels: List[int], num_classes: int = None) -> Dict:
    """
    Compute class distribution statistics.
    
    Useful for understanding dataset imbalance.
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    if num_classes is None:
        num_classes = len(unique)
    
    distribution = {
        'counts': dict(zip(unique.tolist(), counts.tolist())),
        'percentages': dict(zip(unique.tolist(), (counts / len(labels) * 100).tolist())),
        'total': len(labels),
        'num_classes': num_classes,
        'imbalance_ratio': counts.max() / counts.min() if len(counts) > 0 else 1.0
    }
    
    return distribution


def create_balanced_subset(texts: List, labels: List, identities: List, 
                           samples_per_class: int = 1000) -> Tuple:
    """
    Create a balanced subset by sampling equal numbers from each class.
    
    Useful for faster experimentation.
    """
    df = pd.DataFrame({
        'text': texts,
        'label': labels,
        'identity': identities
    })
    
    balanced = df.groupby('label').apply(
        lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=42)
    ).reset_index(drop=True)
    
    return (
        balanced['text'].tolist(),
        balanced['label'].tolist(),
        balanced['identity'].tolist()
    )


if __name__ == '__main__':
    """
    Test data loading functions.
    """
    print("=" * 80)
    print("TESTING DATA LOADERS")
    print("=" * 80)
    
    # Test Jigsaw
    try:
        train_texts, train_labels, train_ids, test_texts, test_labels, test_ids = load_jigsaw_data(sample_size=1000)
        print(f"\n✓ Jigsaw loaded successfully")
        print(f"  Sample text: {train_texts[0][:100]}...")
    except Exception as e:
        print(f"\n✗ Jigsaw loading failed: {e}")
    
    # Test Bias in Bios
    try:
        train_texts, train_labels, train_genders, test_texts, test_labels, test_genders, occupations = load_bias_in_bios()
        print(f"\n✓ Bias in Bios loaded successfully")
        print(f"  Sample text: {train_texts[0][:100]}...")
        print(f"  Occupations: {occupations[:5]}...")
    except Exception as e:
        print(f"\n✗ Bias in Bios loading failed: {e}")
    
    # Test Equity Corpus
    try:
        pairs = load_equity_evaluation_corpus()
        print(f"\n✓ Equity Evaluation Corpus loaded successfully")
        print(f"  Sample pair: {pairs[0]}")
    except Exception as e:
        print(f"\n✗ Equity Corpus loading failed: {e}")
