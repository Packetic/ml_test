import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

ss = SnowballStemmer('russian')

sentence = 'Хочу отреставрировать кухню'
sentence = sentence.lower()

stop_words = set(stopwords.words('russian'))
word_tokens = word_tokenize(sentence)

filtered_sentence = [w for w in word_tokens if w not in stop_words]
stemmed_words = [ss.stem(w) for w in filtered_sentence]
print(stemmed_words)
print(lemmatizer.lemmatize("кактос"))

# TODO: дальше сделать такую штуку: отобрать самые важные слова, пройтись по датасету, а именно строке с названием, если
# TODO: хотя бы одно из важных слов есть в ней, то добавляем ее в список. Список будет подаваться на вход нейронке, а
# TODO: дальше надо придумать что делать со словами, потому что их будет значительно меньше чем строк

# TODO: вторая задача - обучить нейросеть отбирать слова в названиях товаров на самые частовстречаемые категории
