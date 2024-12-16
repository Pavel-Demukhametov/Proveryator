from rutermextract import TermExtractor
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
from typing import List, Optional 
nltk.download('stopwords')

class KeywordsExtractor:
    def __init__(self, top_n: int = 10):
        self.term_extractor = TermExtractor()
        self.stop_words = set(stopwords.words('russian'))
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_vocab = MorphVocab()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.top_n = top_n

    def extract_keywords(self, text: str, top_n: Optional[int] = None) -> List[str]:
        top_n = top_n or self.top_n
        keywords = self.term_extractor(text)[:top_n]
        return [keyword.normalized for keyword in keywords]

    def tokenize_lemmatize(self, text: str) -> str:
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        tokens = []
        for token in doc.tokens:
            if token.pos in {'PUNCT', 'SYM'}:
                continue
            token.lemmatize(self.morph_vocab)
            if token.lemma:
                tokens.append(token.lemma.lower())
            else:
                tokens.append(token.text.lower())

        return ' '.join(tokens)

    def extract_sentences_with_keyword(self, text: str, keyword_lemmatized: str) -> List[str]:
        sentences_with_keyword = []
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)

        for sent in doc.sents:
            lemmatized_sent = self.tokenize_lemmatize(sent.text)
            if keyword_lemmatized in lemmatized_sent:
                sentences_with_keyword.append(sent.text)

        return sentences_with_keyword
