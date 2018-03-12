#
# Import libraries
#
import PyPDF2
import re
import nltk
import glob
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from wordcloud import WordCloud

# Download stopwords and add new words to the list
nltk.download('stopwords')
sw = nltk.corpus.stopwords.words('english')
sw.append("abstract")
sw.append("conclusion")
sw.append("clustercomputœ")
sw.append("thecomputerjournalvolno")
sw.append("")

# Initialize variables and dictionaries to store word data
words = {}
wordFile = {}
isInFile = {}
tfidf = {}
fileCounter = 0


#
# Read from pdf file, word by word
#
def readFromPdf(pdfName):
    global words, wordFile, isInFile
    file = open(pdfName, "rb")
    pdfReader = PyPDF2.PdfFileReader(file)
    for i in range(pdfReader.getNumPages()):
        page = pdfReader.getPage(i).extractText().split()
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


#
# Read from text file, word by word
#
def readFromTxt(txtName):
    global words, wordFile, isInFile
    file = open(txtName)
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


# Get pdf files from the directory and send to readFromPdf function
fileList = glob.glob('*.pdf')
for fileName in fileList:
    readFromPdf(fileName)
    isInFile = dict.fromkeys(isInFile, 0)
    fileCounter += 1

# Get text files from the directory and send to readFromTxt function
fileList = glob.glob('*.txt')
for fileName in fileList:
    readFromTxt(fileName)
    isInFile = dict.fromkeys(isInFile, 0)
    fileCounter += 1

# Calculate tdidf values for each word
for key, value in words.items():
    tfidf[key] = round(value * math.log(fileCounter / wordFile[key]), 2)

# Sort list descending by their values
words = sorted(words.items(), key=lambda x: x[1], reverse=True)
tfidf = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)

# Open csv files to write
tfFile = open('tf_list.csv', 'w')
tfidfFile = open('tfidf_list.csv', 'w')

tfData = ""
tfidfData = ""

# Write inside the csv files
for i in range(50):
    tfFile.write(str(words[i][0]) + '; ' + str(words[i][1]) + '\n')
    tfidfFile.write(str(tfidf[i][0]) + '; ' + str(tfidf[i][1]) + '\n')
    tfData += str(words[i][0]) + " "
    tfidfData += str(tfidf[i][0]) + " "

# Close files
tfFile.close()
tfidfFile.close()

# Settings for wordcloud
plt.figure(figsize=(20, 10))
wordcloud = WordCloud(background_color='white', mode="RGB", width=2000, height=1000).generate(tfData)
plt.title("TF").set_size(35)
plt.imshow(wordcloud)
plt.axis("off")

# Write wordcloud to pdf
with PdfPages('tf_wordCloud.pdf') as pdf:
    pdf.savefig()

# Settings for wordcloud
wordcloud = WordCloud(background_color='white', mode="RGB", width=2000, height=1000).generate(tfidfData)
plt.title("TF-IDF").set_size(35)
plt.imshow(wordcloud)
plt.axis("off")

# Write wordcloud to pdf
with PdfPages('tfidf_wordCloud.pdf') as pdf:
    pdf.savefig()

# Close plot
plt.close()

#
# End of the project
#
