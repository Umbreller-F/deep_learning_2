from config import *
import numpy as np
import os

pad=np.array(
      [ 1.57453286,  0.44730789,  0.02113008,  2.04991462, -1.6387376 ,
        0.0601251 ,  0.78395718, -0.42686316, -0.38119342,  1.12395046,
        0.45047137,  0.18196803,  1.9905336 , -0.11911221, -0.39158346,
       -0.29572857, -0.19221725, -1.82287607, -0.7580225 , -0.25367344,
       -0.82452869, -0.75206921,  1.15031549, -1.05018441, -1.74042713,
       -0.5726877 , -0.67793457, -0.8816091 ,  1.45665465,  1.66210845,
       -0.0763057 , -0.46990705,  0.1538067 , -0.08166611,  0.63377019,
        0.43441394, -1.97347187, -0.41679212, -0.86432106, -1.89221384,
       -1.94871784,  0.65250879, -1.09494552,  0.26285307, -0.9299184 ,
       -0.97699624, -0.01910304, -0.27315189,  1.45246803,  0.28733789,
       -0.68107532,  0.31671158,  1.16217876,  0.99280733, -1.02690349,
       -0.65447401,  1.23105395,  0.1564863 , -0.24363861, -0.11419256,
       -0.40849856,  0.55169219,  0.39842642,  0.74459489,  2.61294403,
        0.69149393,  1.07256174,  0.00585858, -0.37946851,  0.91284637,
       -0.06952849,  1.98191559,  0.08207615, -0.16303484, -0.53623445,
        2.34279301, -3.57435828, -0.0967547 , -0.4136028 , -1.84235293,
        0.93393325, -1.05969761,  1.58730866, -1.67419975,  0.29993341,
       -1.59579608,  1.18277279,  1.08195207, -0.17560604, -0.80528429,
       -0.4978387 ,  0.68523834,  1.08149352, -0.30527368, -0.68328622,
        0.90647836,  1.30607991, -2.13128011, -0.02353469,  1.11826376]
        )

unk=np.array(
      [ 0.70086998, -1.96253374,  1.04667702,  1.00191814,  0.02489337,
        0.4477394 ,  0.9528835 , -1.72805656, -2.4027695 ,  0.90031461,
       -0.64311742,  0.46985492,  0.0649126 , -0.36140919, -1.27277025,
        0.15393009,  0.42667123,  0.08219619, -2.08569436,  0.38875839,
        0.43257594, -1.19014038, -0.18057937,  0.9096513 ,  0.14110975,
       -0.755183  ,  0.37252549, -1.9146299 , -0.13710334,  0.1100864 ,
       -1.34037952, -0.63201702, -0.18971416,  1.37952011, -0.77493629,
       -0.44707888, -0.78791865,  0.32344195, -0.8040147 , -0.92887019,
        0.26026724,  1.25762135, -1.68783328,  0.12224537,  0.52795977,
        1.3987224 , -0.03699951,  0.46472224,  2.98275876,  1.009281  ,
       -0.46055084, -0.95353776, -1.09338816, -0.94087998, -0.95604018,
        0.40219601,  0.26517883,  0.33178817, -0.88603496,  0.3225202 ,
        0.28644212, -1.88109427,  0.02952588, -0.68825886, -0.79778649,
        1.91499623, -0.01243516, -0.26433459,  2.44676878,  0.66083264,
       -0.09012963,  0.94269907,  0.50333274,  0.91305611,  0.68761115,
       -0.19944869,  0.25281423, -0.21434549, -0.73786652,  0.46638957,
       -1.27434803, -1.06681124, -2.75195795, -0.95563356, -1.67053423,
        1.17023417,  0.35816525,  0.9229472 ,  0.81439754,  0.48351118,
       -0.86742924, -0.08728364, -0.63339535, -0.664918  , -0.43059189,
        0.61009962, -0.92694616,  0.05037266,  0.72799334,  0.65589842]
    )


class GloVeWordEmbeddings():
    def __init__(self, glove_file_path, num_dims):
        self.num_dims = num_dims


        if not os.path.exists(glove_file_path):
            print("Error! Not a valid glove path")
            return
        
        self.token_to_embedding = {
            PAD: pad,
            UNK: unk
        }

        with open(glove_file_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                values = line.split()

                # create a dict of word -> positions
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                
                self.token_to_embedding[word] = vector
    
    def get_token_to_embedding(self):
        return self.token_to_embedding
    
    def get_num_dims(self):
        return self.num_dims

    def _get_cosine_similarity(self, vecA: np.array, vecB: np.array):
        return np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))

    def _get_closest_words(self, embedding):
        return sorted(self.token_to_embedding.keys(), key=lambda w: self._get_cosine_similarity(self.token_to_embedding[w], embedding), reverse=True)
    
    def _get_embedding_for_word(self, word: str) -> np.array:
        if word in self.token_to_embedding.keys():
            return self.token_to_embedding[word]
        return np.array([])

    def get_x_closest_words(self, word, num_closest_words=1) -> list: 

        embedding = self._get_embedding_for_word(word)
        if embedding.size == 0:
            print(f"{word} does not exist in the embeddings.")
            return []
        closest_words = self._get_closest_words(embedding)
        for w in [word, PAD, UNK]: closest_words.remove(w)

        return closest_words[:num_closest_words]
    
    def get_word_analogy_closest_words(self, w1, w2, w3, num_closest_words=1):
        e1 = self._get_embedding_for_word(w1)
        e2 = self._get_embedding_for_word(w2)
        e3 = self._get_embedding_for_word(w3)

        if e1.size == 0 or e2.size == 0 or e3.size == 0:
            print(f"{w1}:{e1.size}  {w2}:{e2.size}  {w3}:{e3.size}")
            return []

        embedding = e2 - e1 + e3
        closest_words = self._get_closest_words(embedding)
        for w in [w1, w2, w3, PAD, UNK]: closest_words.remove(w) 
        return closest_words[:num_closest_words]