import os
import pandas as pd
import pickle
from pprint import pprint


def getQueries():
    queryFile = os.getcwd() + '\\Data\\query_desc.51-100.short.txt'
    queries = {}  # {qID: 'query text', ...}

    with open(queryFile, 'r') as qf:
        line = qf.readline()

        while line:
            split = line.strip().split(' ', maxsplit = 1)

            qID = int(split[0][: -1])
            qText = split[1].strip()
            queries[qID] = qText

            line = qf.readline()

    qf.close()
    
    return queries

def getData():
    queries = getQueries()
    qrelDocs = {i: {'rel': set(), 'irr': set()} for i in queries}  # i is query ID
    qrelFile = os.getcwd() + '\\Data\\qrels.adhoc.51-100.AP89.txt'

    # read qrel file and split into rel and irr docs for each query in query file ONLY
    with open(qrelFile, 'r') as qf:
        line = qf.readline()

        while line:
            split = line.strip().split(' ')

            qID, docID, rel = int(split[0]), split[2], int(split[3])
            
            if qID in qrelDocs:
                if rel == 1:
                    qrelDocs[qID]['rel'].add(docID)
                else:
                    qrelDocs[qID]['irr'].add(docID)

            line = qf.readline()

    qf.close()

    print('id irr rel tot')
    for i in qrelDocs:
        print(i, len(qrelDocs[i]['irr']), len(qrelDocs[i]['rel']), len(qrelDocs[i]['irr']) + len(qrelDocs[i]['rel']))

    bm25, minScore_bm25 = readRes(list(qrelDocs.keys()), os.getcwd() + '\\Data\\bm25_unmodq.txt')
    idf, minScore_idf = readRes(list(qrelDocs.keys()), os.getcwd() + '\\Data\\okapi_idf_unmodq.txt')
    otf, minScore_otf = readRes(list(qrelDocs.keys()), os.getcwd() + '\\Data\\okapi_tf_unmodq.txt')
    jm, minScore_jm = readRes(list(qrelDocs.keys()), os.getcwd() + '\\Data\\jm_unmodq.txt')
    laplace, minScore_laplace = readRes(list(qrelDocs.keys()), os.getcwd() + '\\Data\\laplace_unmodq.txt')

    # add irrelevant docs to get 1000 irr docs for each query
    print('\nadding docs to data\n')
    for qID in bm25:  # bm25 had best performance, so add according to its results
        count = 0

        while len(qrelDocs[qID]['irr']) < 1000:
            for docID in bm25[qID]:
                if docID not in qrelDocs[qID]['rel'] and docID not in qrelDocs[qID]['irr']:  # if the doc isn't relevant and isn't already in the irrelevant dict
                    qrelDocs[qID]['irr'].add(docID)  # add doc to irr dict for current query
                    count += 1

                    if len(qrelDocs[qID]['irr']) == 1000:
                        break

    print('id irr rel tot')
    for i in qrelDocs:
        print(i, len(qrelDocs[i]['irr']), len(qrelDocs[i]['rel']), len(qrelDocs[i]['irr']) + len(qrelDocs[i]['rel']))

    ids, bm25Scores, idfScores, otfScores, jmScores, laplaceScores = {}, {}, {}, {}, {}, {}

    for qID in qrelDocs:
        for docID in qrelDocs[qID]['rel']:
            x = str(qID) + '_' + docID
            ids[x] = 1
            
            if docID in bm25[qID]:
                bm25Scores[x] = bm25[qID][docID]
            else:
                bm25Scores[x] = minScore_bm25[qID]

            if docID in idf[qID]:
                idfScores[x] = idf[qID][docID]
            else:
                idfScores[x] = minScore_idf[qID]

            if docID in otf[qID]:
                otfScores[x] = otf[qID][docID]
            else:
                otfScores[x] = minScore_otf[qID]
            
            if docID in jm[qID]:
                jmScores[x] = jm[qID][docID]
            else:
                jmScores[x] = minScore_jm[qID]

            if docID in laplace[qID]:
                laplaceScores[x] = laplace[qID][docID]
            else:
                laplaceScores[x] = minScore_laplace[qID]

        for docID in qrelDocs[qID]['irr']:
            x = str(qID) + '_' + docID
            ids[x] = 0
            
            if docID in bm25[qID]:
                bm25Scores[x] = bm25[qID][docID]
            else:
                bm25Scores[x] = minScore_bm25[qID]

            if docID in idf[qID]:
                idfScores[x] = idf[qID][docID]
            else:
                idfScores[x] = minScore_idf[qID]

            if docID in otf[qID]:
                otfScores[x] = otf[qID][docID]
            else:
                otfScores[x] = minScore_otf[qID]

            if docID in jm[qID]:
                jmScores[x] = jm[qID][docID]
            else:
                jmScores[x] = minScore_jm[qID]

            if docID in laplace[qID]:
                laplaceScores[x] = laplace[qID][docID]
            else:
                laplaceScores[x] = minScore_laplace[qID]

    pickleID = True
    if pickleID:
        with open(os.getcwd() + '\\ids.pickle', 'wb') as pckl:
            pickle.dump(ids, pckl, pickle.HIGHEST_PROTOCOL)
        pckl.close()

    qtfDict, percDict, lenDict = getQTF(list(ids)) 

    df = pd.DataFrame(list(ids.items()), columns = ['id', 'label'])  # create df

    df['qID'] = df['id'].apply(lambda x: int(x.split('_')[0]))  # create label for query by extracting qID from id

    # add scores
    df['bm25'] = df['id'].map(bm25Scores)
    df['idf'] = df['id'].map(idfScores)
    df['otf'] = df['id'].map(otfScores)
    df['jm'] = df['id'].map(jmScores)
    df['laplace'] = df['id'].map(laplaceScores)

    # normalise scores
    df['bm25'] = df.groupby('qID')['bm25'].transform(lambda x: (x - x.mean()) / x.std())
    df['idf'] = df.groupby('qID')['idf'].transform(lambda x: (x - x.mean()) / x.std())
    df['otf'] = df.groupby('qID')['otf'].transform(lambda x: (x - x.mean()) / x.std())
    df['jm'] = df.groupby('qID')['jm'].transform(lambda x: (x - x.mean()) / x.std())
    df['laplace'] = df.groupby('qID')['laplace'].transform(lambda x: (x - x.mean()) / x.std())

    df['qtf'] = df['id'].map(qtfDict)
    df['perc'] = df['id'].map(percDict)
    df['docLen'] = df['id'].map(lenDict)

    pprint(df.head())
    print(df.shape)
    # pprint(df['qID'].value_counts())

    df.to_pickle(os.getcwd() + '\\data.pkl')

    print('*** D O N E  ***')
 
def readRes(qrelDocKeys, resFile):
    # i is query ID for the comprehensions
    res = {i: {} for i in qrelDocKeys}  
    minScore = {i: None for i in qrelDocKeys}

    print('reading', resFile)

    with open(resFile, 'r') as rf:
        line = rf.readline()

        while line:
            split = line.strip().split()

            qID, docID, score = int(split[0]), split[2], float(split[4])

            # if qID in qrelDocKeys:  # remove this condition if not using queries_TEST.txt
            res[qID][docID] = score
            minScore[qID] = score

            line = rf.readline()
    
    rf.close()

    return res, minScore

def getQTF(idList):
    qtfFile = os.getcwd() + '\\Data\\qtf2.txt'
    print('reading qtf')

    tfDict = {i: None for i in idList}  # query term freq (sum of tf of query terms in the doc)
    percDict = {i: None for i in idList}
    lenDict = {i: None for i in idList}

    with open(qtfFile, 'r') as qtf:
        line = qtf.readline()

        while line:
            split = line.strip().split(' ')

            featureID, tf, perc, docLen = split[0] + '_' + split[1], int(split[2]), float(split[3]), int(split[4])

            if featureID in idList:
                tfDict[featureID] = tf
                percDict[featureID] = perc
                lenDict[featureID] = docLen

            line = qtf.readline()

    qtf.close()

    return tfDict, percDict, lenDict

getData()