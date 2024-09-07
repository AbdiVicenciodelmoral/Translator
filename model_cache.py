from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, MarianTokenizer, MarianMTModel

cache_dir = "./models_cache"  # Specify your cache directory

# Wav2Vec2 English model
processor_en = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h", cache_dir=cache_dir)
model_en = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h", cache_dir=cache_dir)

# Wav2Vec2 Spanish model
processor_es = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish", cache_dir=cache_dir)
model_es = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-spanish", cache_dir=cache_dir)

# MarianMT models for translation
tokenizer_es_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-en', cache_dir=cache_dir)
model_es_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-en', cache_dir=cache_dir)

tokenizer_en_es = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-es', cache_dir=cache_dir)
model_en_es = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-es', cache_dir=cache_dir)

tokenizer_es_fr = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-es-fr', cache_dir=cache_dir)
model_es_fr = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-es-fr', cache_dir=cache_dir)