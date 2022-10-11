import neuralcoref
import spacy
import nltk
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sortedcontainers import SortedDict
import importlib.util
import sys
from sklearn.cluster import AgglomerativeClustering
import numpy as np
spec = importlib.util.spec_from_file_location("takahe", "takahe/takahe.py")
takahe = importlib.util.module_from_spec(spec)
sys.modules["takahe"] = takahe
spec.loader.exec_module(takahe)
# print('hi')

def run(file):
    # f=open('transcript.txt','r')
    f = open('static/uploads/' + file)
    transcript=''
    for i in f:
        tem=list(i.lstrip().rstrip().split(':'))
        if len(tem)>1:
            transcript+=tem[1]
        else :transcript+=tem[0]
        transcript+='.'
    transcript=transcript.replace('...',' ').replace('..',' ').replace(',',' ')
    transcript=transcript.replace(')',' ').replace('(',' ').replace('"','').replace('"','')
    transcript

    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    doc1 = nlp(transcript)
    # print(doc1._.coref_resolved)

    fintsc=doc1._.coref_resolved
    embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    corpus=list(fintsc.split('.'))

    corpus_embeddings = embedder.encode(corpus)

    # Then, we perform k-means clustering using sklearn:


    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    corpus_embeddings2 = embedder.encode(corpus)
    # Normalize the embeddings to unit length
    corpus_embeddings2 = corpus_embeddings2 /  np.linalg.norm(corpus_embeddings2, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.9) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings2)
    cluster_assignment = clustering_model.labels_

    clustered_sentences2 = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences2:
            clustered_sentences2[cluster_id] = []

        clustered_sentences2[cluster_id].append(corpus[sentence_id])


    num_clusters = 9
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])

    # for i, cluster in enumerate(clustered_sentences):
        # print("Cluster ", i+1)
        # print(cluster)
        # print("")
    nlist=[]
    for i,cluster in clustered_sentences2.items():
        cflist=[]
        for j in cluster:
            # print(j,'\n')
            fs=''
            wordsList = nltk.word_tokenize(j)
            # print(wordsList)
            tagged = nltk.pos_tag(wordsList)
            # print(tagged)
            for k in tagged:
                s=k[0]
                t=k[1]
                fs=fs+s+'/'+t+' '
            cflist.append(fs)
        nlist.append(cflist)


    stop_words = set(stopwords.words('english'))

    endlist=[]
    for i,cluster in enumerate(clustered_sentences):
        cflist=[]
        for j in cluster:
            # print(j,'\n')
            fs=''
            wordsList = nltk.word_tokenize(j)
            # print(wordsList)
            tagged = nltk.pos_tag(wordsList)
            # print(tagged)
            for k in tagged:
                s=k[0]
                t=k[1]
                fs=fs+s+'/'+t+' '
            cflist.append(fs)
        endlist.append(cflist)
    endlist


    clist=endlist
    summary=open('static/uploads/summarize.txt','w')
    for i in clist:
        sentences=i
        compresser = takahe.word_graph( sentences, 
                                    nb_words = 6, 
                                    lang = 'en', 
                                    punct_tag = "PUNCT" )
        candidates = compresser.get_compression(50)
        d={}
        for cummulative_score, path in candidates:
            normalized_score = cummulative_score / len(path)
            d[round(normalized_score, 3)]=' '.join([u[0] for u in path])
        sl=list(d.keys())
        sl.sort(reverse=True)
        # print(sl)
        if len(sl)>0: summary.write(d[sl[0]]+'\n')
    summary.close()
    clist=nlist
    summary=open('static/uploads/summarize2.txt','w')
    for i in clist:
        sentences=i
        compresser = takahe.word_graph( sentences, 
                                    nb_words = 8, 
                                    lang = 'en', 
                                    punct_tag = "PUNCT" )
        candidates = compresser.get_compression(50)
        d={}
        for cummulative_score, path in candidates:
            normalized_score = cummulative_score / len(path)
            d[round(normalized_score, 3)]=' '.join([u[0] for u in path])
        sl=list(d.keys())
        sl.sort(reverse=True)
        # print(sl)
        if len(sl)>0:summary.write(d[sl[0]]+'\n')
    summary.close()

