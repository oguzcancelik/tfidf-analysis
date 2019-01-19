import glob
import math
import re

import PyPDF2
import matplotlib.pyplot as plt
import nltk
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud


def read_from_pdf(pdf_name):
    global words, wordFile, isInFile
    file = open(pdf_name, "rb")
    pdf_reader = PyPDF2.PdfFileReader(file)
    for i in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(i).extractText().split()
        for j in page:
            temp = re.sub('[-/{}[()!@#£$.,:*"“;><&1234567890]', '', j).lower()
            if temp not in sw and len(temp) > 1:
                if temp in words:
                    words[temp] += 1
                else:
                    words[temp] = 1
                if temp in isInFile and isInFile[temp] == 0:
                    wordFile[temp] += 1
                    isInFile[temp] = 1
                elif temp not in isInFile:
                    wordFile[temp] = 1
                    isInFile[temp] = 1
    file.close()


def read_from_txt(txt_name):
    global words, wordFile, isInFile
    file = open(txt_name)
    for i in file.read().split():
        temp = re.sub('[-/{}[()!@#£$.,:*"“;><&1234567890]', '', i).lower()
        if temp not in sw and len(temp) > 1:
            if temp in words:
                words[temp] += 1
            else:
                words[temp] = 1
            if temp in isInFile and isInFile[temp] == 0:
                wordFile[temp] += 1
                isInFile[temp] = 1
            elif temp not in isInFile:
                wordFile[temp] = 1
                isInFile[temp] = 1
    file.close()


nltk.download('stopwords')
sw = nltk.corpus.stopwords.words('english')
sw += ["abstract", "conclusion", "clustercomputœ", "thecomputerjournalvolno", ""]

words = {}
wordFile = {}
isInFile = {}
tfidf = {}
fileCounter = 0

fileList = glob.glob('*.pdf')
for fileName in fileList:
    read_from_pdf(fileName)
    isInFile = dict.fromkeys(isInFile, 0)
    fileCounter += 1

fileList = glob.glob('*.txt')
for fileName in fileList:
    read_from_txt(fileName)
    isInFile = dict.fromkeys(isInFile, 0)
    fileCounter += 1

for key, value in words.items():
    tfidf[key] = round(value * math.log(fileCounter / wordFile[key]), 2)

words = sorted(words.items(), key=lambda x: x[1], reverse=True)
tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)

tfFile = open('tf_list.csv', 'w')
tfidfFile = open('tfidf_list.csv', 'w')

tfData = ""
tfidfData = ""

for i in range(50):
    tfFile.write(str(words[i][0]) + ', ' + str(words[i][1]) + '\n')
    tfidfFile.write(str(tfidf[i][0]) + ', ' + str(tfidf[i][1]) + '\n')
    tfData += str(words[i][0]) + " "
    tfidfData += str(tfidf[i][0]) + " "

tfFile.close()
tfidfFile.close()

plt.figure(figsize=(20, 10))
wordcloud = WordCloud(background_color='white', mode="RGB", width=2000, height=1000).generate(tfData)
plt.title("TF").set_size(35)
plt.imshow(wordcloud)
plt.axis("off")

with PdfPages('tf_wordCloud.pdf') as pdf:
    pdf.savefig()

wordcloud = WordCloud(background_color='white', mode="RGB", width=2000, height=1000).generate(tfidfData)
plt.title("TF-IDF").set_size(35)
plt.imshow(wordcloud)
plt.axis("off")

with PdfPages('tfidf_wordCloud.pdf') as pdf:
    pdf.savefig()

plt.close()
