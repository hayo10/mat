'''
Extract sentences from Wikipedia
'''

from datasets import load_dataset
import spacy
import numpy as np
from tqdm import tqdm

import utils.pickle as pck
import os
#from aux import file

def formatting_prompts_func(example):
    output_texts = []
    for dt_it in tqdm(range(len(example)), desc='Formatting data into prompts'):
        # text = f"""
        # ### Instruction:
        # Summarize the document in 1 sentence.

        # ### Input:
        # Document: {example[dt_it]["document"]}

        # ### Response:
        # Summary: {example[dt_it]["summary"]}
        # """


        output_texts.append(example[dt_it]['summary'])
    return output_texts


# create missing folders on the way, just a convenience...
def file(*args):
    lst = [arg for arg in args]
    if isinstance(lst[0], list):
        lst = lst[0]
    result = ''
    assert len(lst) >= 2
    if len(lst) > 2:
        folder_to_create = ''
        for folder in lst[:-2]:
            folder_to_create += folder
            try:
                os.mkdir(folder_to_create)
            except FileExistsError:
                pass
            folder_to_create += '/'
    result += ''.join([fld + '/' for fld in lst[:-2]])
    result += lst[-2] + '.' + lst[-1]
    return result

#############################

if __name__ != '__main__':
    raise RuntimeError('This script is not intended to be imported')

#############################
save_file_name = file('experiment', 'sentences',
                      'xsum-sentences', 'pickle')

#############################

xsum = load_dataset('EdinburghNLP/xsum')['train']
xsum_prompts = formatting_prompts_func(xsum)
nlp = spacy.load('en_core_web_sm')

num_of_docs = len(xsum)


def get_sents(doc_id):
    spacyfied_doc = nlp(xsum_prompts[doc_id])
    return [sent for sent in spacyfied_doc.sents]

#############################


doc_ids_already_used = set()

for prompt in tqdm(xsum_prompts, desc='Dumping prompts'):
    pck.dump(prompt, save_file_name)