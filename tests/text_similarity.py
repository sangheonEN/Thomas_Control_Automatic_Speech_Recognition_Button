import difflib
import params
from sentence_transformers import SentenceTransformer
import torch

class Similarity_cal:
    """
    Reference와 Prediction text 간의 유사도를 계산하는 변수와 함수를 포함하는 클래스 
    
    functions:
        1. gestalt_pattern_matching: gestalt_pattern_matching 기반 유사도 측정
        2. sentence_transformers: word2vec 의미론적 유사도 측정 -> 결과가 좋지 않음.
        3. isjunk: 공백 무시 함수
       
    """

    def __init__(self, model_path):
        self.ref_text = [*params.event_flag.keys()]
        """
        hugging face sentence-transformers search url: https://huggingface.co/models?pipeline_tag=sentence-similarity&language=ko&sort=likes
        sentence-transformers pretrained models url: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html?highlight=sentencetransformer%20korea
        1. dragonkue/bge-m3-ko
        2. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        3. jhgan/ko-sroberta-multitask (https://github.com/jhgan00/ko-sentence-transformers, https://huggingface.co/jhgan/ko-sroberta-multitask)
        4. sentence-transformers/distiluse-base-multilingual-cased-v2 (https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)
        5. sentence-transformers/all-MiniLM-L6-v2 (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
        6. sentence-transformers/all-mpnet-base-v2 (https://huggingface.co/sentence-transformers/all-mpnet-base-v2)

        """
        # self.model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        if model_path != None:
            
            self.model = SentenceTransformer(model_path)
            self.model = self.model.to('cuda')
        
        else:
            pass
    
    def isjunk(self, x):
        """
            공백 무시 콜백 함수
            Args:
                x: ??
            Return: x == " "

        """
        return x == " "

    def gestalt_pattern_matching(self, inf_text, threshold):
        """
            Gestalt pattern matching (SequenceMatcher) based similarity.
            threshold default : 0.7
        """

        score_list = []
        for ref_text in self.ref_text:
            similarity = difflib.SequenceMatcher(self.isjunk, inf_text, ref_text).ratio()
            score_list.append(round(similarity, 3))


        max_similarity = max(score_list)
        event_flag = score_list.index(max_similarity) + 1 if max_similarity > threshold else None
        return event_flag, max_similarity, score_list
    
    def sentence_transformers(self, inf_text, threshold):

        embeddings1 = self.model.encode(self.ref_text)
        embeddings2 = self.model.encode(inf_text)
        
        similarities = self.model.similarity(embeddings1, embeddings2)
       
        similarities = torch.round(similarities, decimals=3)

        # print(f"similarities matrix: {similarities}\n")

        max_value, max_index = torch.max(similarities, dim=0)
        max_value = float(max_value)
        max_index = int(max_index) + 1

        event_flag = max_index if max_value > threshold else None
        
        return event_flag, max_value, similarities
    

    def key_word_matching(self, inf_text, threshold):
        """
            How can i key word matching tech?
            threshold default : 0.7
        """
        
        

        pass