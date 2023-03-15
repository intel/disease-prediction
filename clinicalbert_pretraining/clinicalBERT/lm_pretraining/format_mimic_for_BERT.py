import pandas as pd
import sys
import spacy
import re
import argparse
import string

from tqdm import tqdm
from heuristic_tokenize import sent_tokenize_rules 
from os import path, makedirs

root_folder = path.dirname(path.abspath(__file__))
deid_regex  = r"\[\*\*.{0,15}.*?\*\*\]" 

from spacy.language import Language

def preprocess(x):    
    y=re.sub(deid_regex,'',x) 
    y = re.sub('\n', ' ', y)
    y = re.sub('\r', ' ', y)    
    y=re.sub('[0-9]+\.','',y) 
    y=re.sub('dr\.','doctor',y)
    y=re.sub('m\.d\.','md',y)
    y=re.sub('discharge date:','',y)
    y=re.sub('admission date:','',y)
    y=re.sub('--|__|==','',y)
    y = y.translate(str.maketrans("", "", string.digits))
    y = " ".join(y.split())
    
    return y

def data_prep(notes):
    notes['text'] = notes['text'].fillna(' ').str.lower().apply(str.strip)
    notes['text']=notes['text'].apply(lambda x: preprocess(x))

#setting sentence boundaries
@Language.component("component")
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

#convert de-identification text into one token
def fix_deid_tokens(text, processed_text):
    if text:
        indexes = [m.span() for m in re.finditer(deid_regex,text,flags=re.IGNORECASE)]
    else:
        indexes = []
        
    for start,end in indexes:
        processed_text.merge(start_idx=start,end_idx=end)

    return processed_text
    

def process_section(section, note, processed_sections):
    # perform spacy processing on section
    processed_section = nlp(section['sections'])
    processed_section = fix_deid_tokens(section['sections'], processed_section)
    processed_sections.append(processed_section)

def process_note_helper(note):
    # split note into sections
    note_sections = sent_tokenize_rules(note)
    processed_sections = []
    section_frame = pd.DataFrame({'sections':note_sections})
    section_frame.apply(process_section, args=(note,processed_sections,), axis=1)
    return(processed_sections)

def process_text(sent, note):
    sent_text = sent['sents'].text
    if len(sent_text) > 0 and sent_text.strip() != '\n':
        if '\n' in sent_text:
            sent_text = sent_text.replace('\n', ' ')
        note['text'] += sent_text + '\n'  

def get_sentences(processed_section, note):
    # get sentences from spacy processing
    sent_frame = pd.DataFrame({'sents': list(processed_section['sections'].sents)})
    sent_frame.apply(process_text, args=(note,), axis=1)

def process_note(note):
    try:
        note_text = note['text'] #unicode(note['text'])
        note['text'] = ''
        processed_sections = process_note_helper(note_text)
        ps = {'sections': processed_sections}
        ps = pd.DataFrame(ps)
        ps.apply(get_sentences, args=(note,), axis=1)
        return note 
    except Exception as e:
        pass
        #print ('error', e)

def wrapper(output_dir, output_file, mimic_notes_file, category):
    global nlp
    tqdm.pandas()
    print('Begin reading notes')

    notes = pd.read_csv(mimic_notes_file, index_col = 0, low_memory=False)
    if category:
        notes = notes[notes['category'] == category]

    notes.columns = [i.lower() for i in notes.columns]
    data_prep(notes)

    print('Number of notes: %d' %len(notes.index))
    notes['ind'] = list(range(len(notes.index)))
    
    nlp = spacy.load('en_core_sci_md', disable=['tagger','ner', 'lemmatizer'])
    nlp.add_pipe("component", before='parser')  

    formatted_notes = notes.progress_apply(process_note,axis=1)
    makedirs(output_dir, exist_ok = True) 
    
    with open(output_dir  + output_file,'w') as f:
        for text in formatted_notes['text']:
            if text != None and len(text) != 0 :
                f.write(text)
                f.write('\n')

    print ("Done formatting notes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This formats MIMIC data for clinicalBERT"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", default=path.join(root_folder,'../../outputs/tokenized_notes/') 
    )

    parser.add_argument(
        "--output_file", type=str, help="Output file", default='formatted_output.txt'
    )

    parser.add_argument(
         "--mimic_notes_file", type=str, help="MIMIC Notes File", default=path.join(root_folder,'../../data/NOTEEVENTS.csv')
    )

    parser.add_argument(
         "--category", type=str, help="Sub-Category Name", default=None
    )

    params = parser.parse_args()

    wrapper(params.output_dir, params.output_file, params.mimic_notes_file, params.category)
