from pipelines import pipeline as qa_pipeline
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import nltk
from googletrans import Translator
from sklearn.metrics.pairwise import cosine_similarity
import re

nltk.download('punkt')
nltk.download('stopwords')

class qaMatcher:
    
    def __init__(self, model_ans= "deepset/roberta-base-squad2"):
        self.translator = Translator()
        self.gen_question = qa_pipeline("e2e-qg")
        self.gen_answer = pipeline('question-answering', model=model_ans, tokenizer=model_ans)
        self.model_sim = SentenceTransformer('all-mpnet-base-v2')
        
    def translate(self, text, to='en', src='ru'):
        translated_text = self.translator.translate(text, src=src, dest=to).text
        return translated_text
    
    def generate_questions(self, text):
        questions = []
        for sent in text.split('.'):
            questions.extend(self.gen_question(sent.strip()))
        return questions
    
    def answer(self, questions, context, conf_limit=0.3):

        result = []
        for question in questions:
            obj = self.gen_answer(question=question, context=context)
            ans = obj['answer']
            conf = obj['score']
            if conf > conf_limit:
                result.append(ans)
            else:
                result.append(None)
        return result
    
    def texts_similarity(self, text1, text2):
        '''Сравнение текстов (ответов) по идентичности, возвращает значение [0, 1]'''
        vecs = self.model_sim.encode([text1, text2])
        sim = cosine_similarity(vecs)[0][1]
        sim = (sim + 1) / 2
        return sim
    
    def text_preprocess(self, DOCUMENT):
        DOCUMENT = re.sub(r'\n|\r', ' ', DOCUMENT)
        DOCUMENT = re.sub(r' +', ' ', DOCUMENT)
        DOCUMENT = DOCUMENT.strip()
        return DOCUMENT
    
    def predict(self, orig_text, text,  ans_conf=1e-4, score_conf=0.5):
        text = self.text_preprocess(text)
        orig_text = self.text_preprocess(orig_text)

        text_en = self.translate(text)
        orig_text_en = self.translate(orig_text)

        # Генерируем вопросы для дальнейшей проверки
        text_questions = self.generate_questions(text_en)

        text_answers = self.answer(text_questions, text_en, conf_limit=ans_conf)

        # Отвечаем на вопросы по найденной статье
        text_ans = self.answer(text_questions, orig_text_en, conf_limit=ans_conf)

        # Сравниваем ответы
        results = []
        total_score = 0.
        counter = 0
        for question, true_ans, ans_to_check in zip(text_questions, text_ans, text_answers):
            if true_ans == None or ans_to_check == None:
                continue

            counter += 1
            score = self.texts_similarity(true_ans, ans_to_check) # возвращает значение похожести ответов от [0, 1]
            total_score += score
            results.append({
                'question': self.translate(question, to='ru', src='en'),
                'true_answer': self.translate(true_ans, to='ru', src='en'),
                'answer_from_article': self.translate(ans_to_check, to='ru', src='en'),
                'confidence': 1-score,
                'matching': score >= score_conf
            })

        results.sort(key=lambda el: el['confidence'])
        max_fake_conf = 1 - results[0]['confidence']
        doubtful_results = [el for el in results if el['matching']]
        fake_score = 1 - total_score / counter
        
        main_info = {
            'max_fake_score' : max_fake_conf,
            'mean_fake_score' : fake_score,
            
        }
        
        return doubtful_results, main_info
    