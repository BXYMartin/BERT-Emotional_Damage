from transformers import MarianMTModel, MarianTokenizer
import tqdm
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


class BackTranslate:
    """
    Language	French	Spanish	Italian	Portuguese	Romanian	Catalan	Galician	Latin
    Code	    fr	    es	    it	    pt	        ro	        ca	    gl	        la
    """
    batch_size = 16

    def __init__(self):
        self.target_model_name = 'Helsinki-NLP/opus-mt-en-ROMANCE'
        self.target_tokenizer = MarianTokenizer.from_pretrained(self.target_model_name)
        self.target_model = MarianMTModel.from_pretrained(self.target_model_name)
        self.en_model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
        self.en_tokenizer = MarianTokenizer.from_pretrained(self.en_model_name)
        self.en_model = MarianMTModel.from_pretrained(self.en_model_name)
        self.target_model.cuda()
        self.en_model.cuda()

    def translate(self, texts, model, tokenizer, language="fr"):
        # Prepare the text data into appropriate format for the model
        template = lambda text: f"{text}" if language == "en" else f">>{language}<< {text}"
        src_texts = [template(text) for text in texts]

        self.text_loader = DataLoader(src_texts,
                                       sampler=SequentialSampler(src_texts),
                                       batch_size=self.batch_size)
        translated_texts = []
        with tqdm.tqdm(self.text_loader, unit="batch") as tepoch:
            for i, data in enumerate(tepoch):
                # Tokenize the texts
                encoded = tokenizer(data, return_tensors="pt", padding=True, max_length=512, truncation=True).to("cuda")
                # Generate translation using model
                translated = model.generate(**encoded)
                #translated = model.generate(torch.tensor(encoded['input_ids']).cuda(),
                #           attention_mask=torch.tensor(encoded['attention_mask']).cuda())

                # Convert the generated tokens indices back into text
                output = tokenizer.batch_decode(translated, skip_special_tokens=True)
                translated_texts.extend(output)
                tepoch.set_description(f"Processing {i}")

        return translated_texts

    def back_translate(self, texts, source_lang="en", target_lang="fr"):
        # Translate from source to target language
        fr_texts = self.translate(texts, self.target_model, self.target_tokenizer,
                             language=target_lang)

        # Translate from target language back to source language
        back_translated_texts = self.translate(fr_texts, self.en_model, self.en_tokenizer,
                                          language=source_lang)

        return back_translated_texts
