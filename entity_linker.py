import sys
import os
import spacy
import pandas as pd
import json

# Add datasets/code to the python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'experiment_datasets', 'code'))

from hotpot_loader import load_hotpot_qa
from tabulate import tabulate


def entity_linker():
    """Initializes the spaCy pipeline with entity linker."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Model not found. Please run 'python -m spacy download en_core_web_sm'")
        return None
    
    # Add entity linker pipeline
    # The 'entityLinker' component is added by spacy-entity-linker
    nlp.add_pipe("entityLinker", last=True)
    return nlp

def extract_wikidata_ids(text, nlp):
    """
    Extracts Wikidata IDs from text using the provided spaCy NLP pipeline.
    
    Args:
        text (str): The input text to process.
        nlp: The spaCy NLP object.
        
    Returns:
        list: A list of dictionaries containing entity information.
    """
    doc = nlp(text)
    entities = []
    
    # spacy-entity-linker adds 'linkedEntities' extension to the doc
    # But usually it's accessed via doc._.linkedEntities
    
    if doc._.linkedEntities:
        for entity in doc._.linkedEntities:
            entities.append({
                "text": entity.get_span().text,
                "wikidata_id": "Q" + str(entity.get_id()),
                "label": entity.get_label(),
                "description": entity.get_description()
            })
            
    return entities

def main():
    
    # Load HotpotQA dataset using the loader module
    dataset = load_hotpot_qa(split="train", subset="distractor", streaming=True)
    
    nlp = entity_linker()
    if not nlp:
        return

    results = []
    
    print("Processing samples...")
    # Process first 5 examples
    for i, example in enumerate(dataset):
        if i >= 5:
            break
            
        question = example['question']
        print(f"Processing question {i+1}: {question}")
        
        extracted_entities = extract_wikidata_ids(question, nlp)
        
        for entity in extracted_entities:
            results.append({
                "question_id": example['id'],
                "original_text": question,
                "entity_text": entity['text'],
                "wikidata_id": entity['wikidata_id'],
                "description": entity['description'],
                "label": entity['label']
            })
            
    # Convert to DataFrame and save/display
    df = pd.DataFrame(results)
    
    if not df.empty:
        print("\nExtracted Entities:")

        display_df = df[['original_text', 'entity_text', 'wikidata_id', 'description']].copy()
        display_df['original_text'] = display_df['original_text'].apply(
            lambda x: (x[:50] + '...') if len(x) > 50 else x)
        display_df['description'] = display_df['description'].apply(
            lambda x: (x[:40] + '...') if len(str(x)) > 40 else x)
            
        print(tabulate(display_df, headers='keys', tablefmt='pretty'))

        output_file = os.path.join(os.path.dirname(__file__), 'experiment_datasets', 'extracted_wikidata_ids.json')
        df.to_json(output_file, orient="records", indent=2)
        print(f"\nResults saved to {output_file}")
    else:
        print("\nNo entities extacted.")

if __name__ == "__main__":
    main()
