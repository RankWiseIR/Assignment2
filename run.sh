#!/bin/bash

source /speech/arjun/miniconda3/bin/activate
conda activate nlp

python main.py \
    -dataset cranfield/ \
    -out_folder output/ \
    -segmenter punkt \
    -tokenizer ptb

# python main.py \
#     -dataset /speech/arjun/exps/1study/CS6370-NLP/assignment1/cranfield/ \
#     -out_folder /speech/arjun/exps/1study/CS6370-NLP/Project/template_code_part2/template_code_part2/output/ \
#     -segmenter punkt \
#     -tokenizer ptb \
#     -custom