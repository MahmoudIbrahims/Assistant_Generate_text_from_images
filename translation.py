from deep_translator import GoogleTranslator

def transArabic(text):
    translate =GoogleTranslator(source='en', target='ar').translate(text)
    return translate



def transEnglish(text):
    translate =GoogleTranslator(source='ar', target='en').translate(text)
    return translate















