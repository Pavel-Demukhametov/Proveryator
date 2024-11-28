from transformers import AutoTokenizer, T5ForConditionalGeneration
from rutermextract import TermExtractor
from functools import partial
from natasha import Doc, Segmenter, NewsEmbedding, MorphVocab, NewsMorphTagger
import nltk
from nltk.corpus import stopwords
import torch
nltk.download('stopwords')

def process_text(text):
    term_extractor = TermExtractor()
    keywords = term_extractor(text)[:10]
    for keyword in keywords:
        print(keyword.normalized)

    print()

    model_name = 'hivaze/AAQG-QA-QG-FRED-T5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

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

    generate_text = partial(generate_text, tokenizer=tokenizer, model=model)

    AAQG_PROMPT = "Сгенерируй вопрос по тексту, используя известный ответ. Текст: '{context}'. Ответ: '{answer}'."

    stop_words = set(stopwords.words('russian'))
    punct = {',', '?', '!', '.', ';', ':', '...', '(', ')', '\'', '\"', '«', '»'}
    segmenter = Segmenter()
    emb = NewsEmbedding()
    morph_vocab = MorphVocab()
    morph_tagger = NewsMorphTagger(emb)

    def tokenize_lemmatize(text):
        doc = Doc(text)
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)

        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            if token.lemma:
                token.text = token.lemma

        return ' '.join(token.text for token in doc.tokens)

    doc = Doc(text)
    doc.segment(segmenter)

    for keyword in keywords:
        keyword_combined = ' '.join(
            word.get_word() for word in keyword.words
        )
        keyword_lemmatized = tokenize_lemmatize(keyword_combined)

        for sent in doc.sents:

            if keyword_lemmatized in tokenize_lemmatize(sent.text):

                generated = generate_text(AAQG_PROMPT.format(
                    context=sent.text,
                    answer=keyword.normalized
                ), n=1)

                print(f'Ответ: {keyword_lemmatized}')
                print(f'Предложение: {sent.text}')
                print(f'Вопрос: {generated}')
                print()

