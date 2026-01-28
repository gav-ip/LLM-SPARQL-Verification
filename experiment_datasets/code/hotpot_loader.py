from datasets import load_dataset

def load_hotpot_qa(split="train", subset="distractor", streaming=True):
    """
    Loads the HotpotQA dataset from Hugging Face.
    
    Args:
        split (str): The split to load (e.g., 'train', 'validation').
        subset (str): The configuration name (e.g., 'distractor', 'fullwiki').
        streaming (bool): Whether to stream the dataset.
        
    Returns:
        Dataset: The loaded dataset.
    """
    print(f"Loading HotpotQA dataset ({subset}, {split})...")
    try:
        dataset = load_dataset("hotpot_qa", subset, split=split, streaming=streaming)
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
