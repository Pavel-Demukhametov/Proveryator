from transformers import AutoTokenizer, T5ForConditionalGeneration
from rutermextract import TermExtractor
from functools import partial
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
import torch
nltk.download('stopwords')

def extract_keywords(text):
    term_extractor = TermExtractor()
    keywords = term_extractor(text)[:10]
    return keywords

def load_model_and_tokenizer(model_name='hivaze/AAQG-QA-QG-FRED-T5-large'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return tokenizer, model, device

def generate_text(prompt, tokenizer, model, n=1, temperature=0.8, num_beams=3):
    encoded_input = tokenizer.encode_plus(prompt, return_tensors='pt')
    encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}

    resulted_tokens = model.generate(**encoded_input,
                                      max_new_tokens=64,
                                      do_sample=True,
                                      num_beams=num_beams,
                                      num_return_sequences=n,
                                      temperature=temperature,
                                      top_p=0.9,
                                      top_k=50)

    resulted_texts = tokenizer.batch_decode(resulted_tokens, skip_special_tokens=True)
    return resulted_texts

def tokenize_lemmatize(text, segmenter, morph_tagger, morph_vocab):
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
        if token.lemma:
            token.text = token.lemma

    return ' '.join(token.text for token in doc.tokens)

def extract_sentences_with_keyword(doc, keyword_lemmatized, segmenter, morph_tagger, morph_vocab):
    sentences_with_keyword = []
    for sent in doc.sents:
        if keyword_lemmatized in tokenize_lemmatize(sent.text, segmenter, morph_tagger, morph_vocab):
            sentences_with_keyword.append(sent.text)
    return sentences_with_keyword

def process_text(text):
    keywords = extract_keywords(text)

    model_name = 'hivaze/AAQG-QA-QG-FRED-T5-large'
    tokenizer, model, device = load_model_and_tokenizer(model_name)

    stop_words = set(stopwords.words('russian'))
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_vocab = MorphVocab()
    morph_tagger = NewsMorphTagger(emb)

    AAQG_PROMPT = "Сгенерируй вопрос по тексту, используя известный ответ. Текст: '{context}'. Ответ: '{answer}'."
    
    doc = Doc(text)
    doc.segment(segmenter)
    for keyword in keywords:
        keyword_combined = ' '.join(word.get_word() for word in keyword.words)
        keyword_lemmatized = tokenize_lemmatize(keyword_combined, segmenter, morph_tagger, morph_vocab)

        sentences_with_keyword = extract_sentences_with_keyword(doc, keyword_lemmatized, segmenter, morph_tagger, morph_vocab)
        for sentence in sentences_with_keyword:
            generated = generate_text(AAQG_PROMPT.format(
                context=sentence,
                answer=keyword.normalized
            ), tokenizer=tokenizer, model=model, n=1)

            print(f'Ответ: {keyword_lemmatized}')
            print(f'Предложение: {sentence}')
            print(f'Вопрос: {generated}')
            print()
