from CNLTK.learner import Seq2SeqLearner
import pandas as pd
#from cntk.learner import Seq2SeqLearner
# Step 1
# Load the downstream data and convert it to Pandas DataFrame.
# The DataFrame needs to contrain both 'source' and 'target' columns.
class predictbaseline:
    def __init__(self) -> None:
        self.learner = Seq2SeqLearner(model_uri='/home/ubuntu/nlg-marketing/model/finetuned')
    def load_data(self,name):
        data_path = "/home/ubuntu/nlg-marketing/data/processed"
        with open(f'{data_path}/{name}.src', 'r', encoding='utf-8') as f:
            src_lines = f.readlines()
        with open(f'{data_path}/{name}.tgt', 'r', encoding='utf-8') as f:
            tgt_lines = f.readlines()
        list_of_tuples = list(zip(src_lines, tgt_lines))
        data = pd.DataFrame(list_of_tuples, columns=['source', 'target'])
        return data
    def predict(self,pretxt):
    #test_data = load_data('test')
    #test_source_list = test_data.head(5).source.tolist()
    #print(test_source_list)
    # Step 3
    # Predic results

    # You can load a specific checkpoint with the model_uri parameter.
    #learner = Seq2SeqLearner(model_uri='/home/ubuntu/nlg-marketing/code/outputs/20210803-1446/checkpoint-24000') 
        generate_list =  self.learner.predict([pretxt],
                    min_length=10,
                    max_length=256,
                    do_sample=True,
                    top_k=10,
                    top_p=0.95,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3
        )
        return generate_list