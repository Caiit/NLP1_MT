# NLP1_MT
Machine Translation Model project for NLP1


Translate
` python3 translate.py -model ../models_project6/ted_sgd_acc_55.43_ppl_12.39_e11.pt -src ../data/ted/valid_eng.txt -output ../output/ted/valid_pred.txt -replace_unk -verbose -pickle ../output/ted/valid_attn.pkl `

Beer
` ./beer -s ../output/ted/valid_pred.txt -r ../data/ted/valid_nld.txt --printSentScores`
