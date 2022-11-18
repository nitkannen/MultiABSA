import json

data_splits = ['train_convert.json', 'dev_convert.json', 'test_convert.json']

for split in data_splits:
    file = open(split)
    data = json.load(file)

    ### Aspect Extraction

    ae_text = []
    oe_text = []
    pair_text = []
    aesc_text = []
    
    for sample in data:
        asp = []
        ops = []
        sentis = []
        pairs = []
        aescs = []
        for aspect in sample['aspects']:
            asp.append(' '.join(aspect['term']).strip())
            sentis.append(aspect['polarity'])

        for opinion in sample['opinions']:
            ops.append(' '.join(opinion['term']).strip())

        ae_text.append(' | '.join(asp) + '\n')
        oe_text.append(' | '.join(ops) + '\n')

        assert len(ops) == len(asp)
        assert len(asp) == len(sentis)

        for aspect, opinion in zip(asp, ops):
            p = [aspect, opinion]
            pairs.append(' ; '.join(p))
        
        for aspect, sentiment in zip(asp, sentis):
            p = [aspect, sentiment]
            aescs.append(' ; '.join(p))

        pair_text.append(' | '.join(pairs) + '\n')
        aesc_text.append(' | '.join(aescs) + '\n')

    file_prefix = split.split('_')[0]
    print(ae_text)
    ae_file = open(f"{file_prefix}.ae","w")
    ae_file.writelines(ae_text)
    ae_file.close()

    oe_file = open(f"{file_prefix}.oe","w")
    oe_file.writelines(oe_text)
    oe_file.close()

    pair_file = open(f"{file_prefix}.pair","w")
    pair_file.writelines(pair_text)
    pair_file.close()

    aesc_file = open(f"{file_prefix}.aesc","w")
    aesc_file.writelines(pair_text)
    aesc_file.close()






        




    ## Opin Extraction


    ### Pair Extraction (AO)


    ## Aspect And Sentiment Extraction


    ## Aspect Level Opinion Extraction


    ## Aspect Level Sentiment Extraction
