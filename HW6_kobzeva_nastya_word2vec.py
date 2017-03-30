import os
import pandas as pd
import numpy as np
import pymorphy2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings(action='ignore',
                        category=UserWarning, module='gensim')
import sys
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

m = 'ruwikiruscorpora_0_300_20.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=True)
model.init_sims(replace=True)

d = {'ADJF':'ADJ', # pymorphy2: word2vec
'ADJS' : 'ADJ',
'ADVB' : 'ADV',
'COMP' : 'ADV',
'CONJ' : 'CCONJ',
'GRND' : 'VERB',
'INFN' : 'VERB',
'INTJ' : 'INTJ',
'LATN' : 'X',
'NOUN' : 'NOUN',
'NPRO' : 'PRON',
'NUMB' : 'NUM',
'NUMR' : 'NUM',
'PNCT' : 'PUNCT' ,
'PRCL' : 'PART',
'PRED' : 'ADV',
'PREP' : 'ADP',
'PRTF' : 'ADJ',
'PRTS' : 'VERB',
'ROMN' : 'X',
'SYMB' : 'SYM',
'UNKN' : 'X',
'VERB' : 'VERB'}


def files_open(path): # обходим все файлы в папке
    for root, dirs, files in os.walk(path):
        for filename in files:
            fname = os.path.join(root, filename)
            with open(fname, 'r', encoding='utf-8') as f:
                yield f.read()

                
morph = pymorphy2.MorphAnalyzer()
def preprocess(word): # приводим к формату слово_POS
    if word.isalpha():
        analysis = morph.parse(word)[0]
        if analysis != []:
            lemma = analysis.normal_form
            tag = analysis.tag.POS
            if tag is not None:
                if tag in d:
                    POS = d[tag]
                    word_POS = lemma + '_' + POS
                    return word_POS


def get_vectors(path):
    texts_vectors = []
    for file in files_open(path):
        words = file.split()
        vectors = []
        for word in words:
            word = word.strip(',.?^&*()-_#{}[]\\—<>!@"\';:/%').lower()
            word_POS = preprocess(word)
            if word_POS in model: # ищем слово в модели
                ##print(word_POS)
                vector = model[word_POS]
                vectors.append(vector)
        text_vector = np.mean(np.array(vectors), axis=0)# средний вектор слов
        texts_vectors.append(text_vector) # вектора всех текстов
    return texts_vectors


def main():
    anekdots = get_vectors('./anekdots')
    izvest = get_vectors('./izvest')
    teh_mol = get_vectors('./teh_mol')
    # создаем датафрейм
    vectors = [i for i in anekdots] + [i for i in izvest] + [i for i in teh_mol]
    df = pd.DataFrame.from_records(vectors)
    df['class'] = ['anekdot' for _ in range(125)] + ['izvest' for _ in range(125)] + ['teh_mol' for _ in range(125)]
    # перемешиваю вектора с помощью индексов строк
    df = df.reindex(np.random.permutation(df.index))
    df.to_csv('./vectors.csv', index=False) # записываю в csv без индексов
    df = pd.read_csv('./vectors.csv')
    #print(df.head())
    X, y = df.iloc[:,:300], df['class']
    # с кросс-валидацией ну слишком долго
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    logistic_model = LogisticRegression() 
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)
    print(classification_report(y_test, y_pred)) 


if __name__ == '__main__':
    main()
            
    
