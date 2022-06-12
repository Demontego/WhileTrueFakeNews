from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,

    Doc
)
import string
import json
from transformers import AutoTokenizer, AutoModel
import torch
import re
import numpy as np
from ner_model import NERWrapper

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity as cos
    
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class MatchTitle:
    
    def __init__(self, data, ngram=3, k_matches=10):
        self.data = data
        lookup = data.title.str.lower().to_list()
        self.vectorizer = TfidfVectorizer(min_df=1, analyzer='char', ngram_range=(1,ngram))
        tf_idf_lookup = self.vectorizer.fit_transform(lookup)
        self.nbrs = NearestNeighbors(n_neighbors=k_matches, n_jobs=-1, metric="cosine").fit(
            tf_idf_lookup
        )
    def get_orig_title(self, original, k_matches=5):
        tf_idf_original = self.vectorizer.transform([original.lower()])
        distances, lookup_indices = self.nbrs.kneighbors(tf_idf_original)
        lookups = self.data.iloc[lookup_indices[0]][['id','title','date', 'text']].copy()
        lookups['site'] = lookups['id'].apply(lambda x: f'https://www.mos.ru/news/item/{x}')
        lookups['sim_title'] = 1 - distances[0]
            
        return lookups.drop_duplicates('id').iloc[:k_matches]


class MatcherText:
    
    def __init__(self, name_model="IlyaGusev/news_tg_rubert"):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        self.names_extractor = NamesExtractor(self.morph_vocab)
        
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        self.model = AutoModel.from_pretrained(name_model)
        self.model.eval()
        
        
    def _get_embs(self, text):
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        doc.parse_syntax(self.syntax_parser)
        doc.tag_ner(self.ner_tagger)
        for span in doc.spans:
            span.normalize(self.morph_vocab)
            
        embs={}
        
        for sent in doc.sents:
            if len(sent.spans)==0:
                continue
            txt = sent.text
            inputs = self.tokenizer(txt, padding=True, truncation=True, max_length=24, return_tensors='pt')
            outputs = self.model(**inputs)
            sentence_emb = mean_pooling(outputs, inputs['attention_mask']).detach().cpu().numpy()
            for span in sent.spans:
                embs[span.normal] = embs[span.normal] + sentence_emb if span.normal in embs else sentence_emb
        return embs
            
        
    def get_score_fake(self, text, orig_text):
        
        orig_emb = self._get_embs(orig_text)
        if isinstance(text, list):
            embs = []
            for t in text:
                embs.append(self._get_embs(t))
        else:
            embs = [self._get_embs(text)]
        
        scores = []
        for emb in embs:
            sim = []
            for key in orig_emb.keys() & emb.keys():
                sim.append(cos(orig_emb[key], emb[key])[0,0])
            coef = len(orig_emb.keys() & emb.keys())/len(orig_emb.keys())
            if len(sim)==0:
                sim.append(0)
            scores.append(coef*np.mean(sim))
            
        return np.array(scores)
    
    
class upMatcherText:
    
    def __init__(self, name_model="IlyaGusev/news_tg_rubert", ner_ckpt="NER_roberta_base_uncased.pck",
                 device='cuda:0'):
        
        self.ner_tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
        with open('ner_labels.dict') as f:
            ner_dict = json.load(f)
        self.inv_ner_dict = {k : v for v, k in ner_dict.items()}
        self.device = device
        self.ner_model = NERWrapper(len(ner_dict), 1000)
        self.ner_model.load_state_dict(torch.load(ner_ckpt, map_location='cpu'))
        self.ner_model.to(device)
        self.ner_model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(name_model)
        self.model = AutoModel.from_pretrained(name_model)
        self.model.eval()
        
    def _kill_punct(self, text):
        killed = text.strip(string.punctuation + string.whitespace)
        if killed[-1] in string.punctuation + string.whitespace or killed[0] in string.punctuation + string.whitespace:
            killed = self._kill_punct(killed)
        return killed
    
    def _clean_text(self, text):
        return re.sub(r'\\n', ' ', re.sub(r'\s+', ' ', text))
        
    def _get_ners(self, sent):
        sent = self._clean_text(sent)
        texts = [sent]
        to_ner = [sent]
        toked = self.ner_tokenizer(to_ner, add_special_tokens=True, return_tensors='pt', padding=True,
                            truncation=True, max_length=512)
        inp_id = toked.input_ids.to(self.device)
        att_mask = toked.attention_mask.to(self.device)
        with torch.no_grad():
            out = self.ner_model.forward(inp_id, att_mask)
        labels = out.argmax(-1).cpu()
        candidates = []
        for i, (lab) in enumerate(labels.tolist()):
            pos = 0
            tmp_words = []
            tmp_labels = []
            previous_word_idx = None
            for l, idx in zip(lab, toked.word_ids(i)):
                if idx is None:
                    previous_word_idx = idx
                    continue
                elif idx != previous_word_idx:
                    w_pos = toked.word_to_chars(i, pos)
                    pos += 1
                    new_w = texts[i][w_pos.start : w_pos.end]
                    tmp_words.append(new_w)
                    tmp_labels.append(self.inv_ner_dict[l])
                previous_word_idx = idx
            pr_l = ''
            cur_cand = ''
            cur_tag = ''
            for w, l in zip(tmp_words, tmp_labels):
                if l == 'O':
                    if len(cur_cand) != 0:
                        if len(self._kill_punct(cur_cand)) > 0:
                            candidates.append((self._kill_punct(cur_cand), cur_tag))
                        cur_cand = ''
                        pr_l = l
                        cur_tag = l
                elif l.startswith('B-'):
                    if len(cur_cand) != 0 and len(self._kill_punct(cur_cand)) > 0:
                        candidates.append((self._kill_punct(cur_cand), cur_tag))
                    cur_cand = w
                    pr_l = l
                    cur_tag = l[2:]
                elif l.startswith('I-'):
                    if pr_l.startswith('B-') and (pr_l[2:] == l[2:]):
                        cur_cand += ' ' + w
                    else:
                        if len(cur_cand) != 0 and len(self._kill_punct(cur_cand)) > 0:
                            candidates.append((self._kill_punct(cur_cand), cur_tag))
                        cur_cand = w
                        pr_l = l
                        cur_tag = l[2:]
            if len(cur_cand) != 0 and len(self._kill_punct(cur_cand)) > 0:
                candidates.append((self._kill_punct(cur_cand), cur_tag))
        return candidates
        
        
    
        
    def _get_embs(self, text):
        embs={}
        for sent in text.split('.'):
            spans = self._get_ners(sent)
            if len(spans)==0:
                continue
            inputs = self.tokenizer(sent, padding=True, truncation=True, max_length=24, return_tensors='pt')
            outputs = self.model(**inputs)
            sentence_emb = mean_pooling(outputs, inputs['attention_mask']).detach().cpu().numpy()
            for span in spans:
                embs[span[0]] = embs[span[0]] + sentence_emb if span[0] in embs else sentence_emb
        return embs
            
        
    def get_score_fake(self, text, orig_text):
        
        orig_emb = self._get_embs(orig_text)
        if isinstance(text, list):
            embs = []
            for t in text:
                embs.append(self._get_embs(t))
        else:
            embs = [self._get_embs(text)]
        
        scores = []
        for emb in embs:
            sim = []
            for key in orig_emb.keys() & emb.keys():
                sim.append(cos(orig_emb[key], emb[key])[0,0])
            coef = len(orig_emb.keys() & emb.keys())/len(orig_emb.keys())
            if len(sim)==0:
                sim.append(0)
            scores.append(coef*np.mean(sim))
            
        return np.array(scores)