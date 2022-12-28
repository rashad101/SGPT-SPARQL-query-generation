from utils.metrics import (
    UnigramMetric, NGramDiversity,
    CorpusNGramDiversity,
    BLEU, METEOR, ROUGE, SPBLEU, SPUnigramMetric
)
import argparse
from os.path import join
import json
from tqdm import tqdm

def is_num(text):
    """ Check if the string is a number"""
    text = text.replace("\'","")
    text = text.replace('\"', "")
    return text.replace('.','',1).isdigit()

def adjust(query):
    ref_vars = list()
    cleaned_query = query
    for token in query.split():
        if token.startswith("?") and token not in ref_vars:
            ref_vars.append(token)

    for i,var in enumerate(ref_vars):
        cleaned_query = cleaned_query.replace(var,f"var{i+1}")
    return cleaned_query

def process_output(text, args):
    kewords = ["(","{",")","}",",",]
    if args.masked:
        kewords+=["ent1", "ent2", "ent3","ent4"]
    for kw in kewords:
        text = text.replace(kw, " " + kw + " ")
    text = text.replace("  ", " ")
    text = " ".join(w if ":" in w else w.lower() for w in text.split(" ")).replace("  "," ")
    text = text.replace("?", " ?").replace("  ", " ")
    out = " "
    for w in text.split(" "):
        if "." in w:
            if is_num(w):
                out += " " + w
            else:
                out += w.replace(".", " .")
        else:
            out+= " "+w
    kw = ['dbo:','dbp:','dbpedia2:','yago:','foaf:','onto:','res:','dbr:','dbc:','wd:','wdt:','ps:', 'pq:']
    for k in kw:
        text = " ".join( w.replace(k," "+k) if k in w else w for w in out.split(" ")).replace("  "," ")
    for k in kw:
        text = text.replace(k," "+k).replace("  "," ")

    return text.replace("  ", " ").strip()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="Output file")
    parser.add_argument('--masked', action='store_true')
    args = parser.parse_args()


    data = json.load(open(join("outputs", args.filepath)))
    em = 0
    em_adj = 0
    metrics = [
        UnigramMetric(metric="recall"),
        UnigramMetric(metric="precision"),
        SPUnigramMetric(metric="recall"),
        SPUnigramMetric(metric="precision"),
        NGramDiversity(n=1),
        NGramDiversity(n=2),
        NGramDiversity(n=3),
        NGramDiversity(n=4),
        CorpusNGramDiversity(n=1),
        CorpusNGramDiversity(n=2),
        CorpusNGramDiversity(n=3),
        CorpusNGramDiversity(n=4),
        BLEU(),
        SPBLEU(),
        METEOR(),
        ROUGE()
    ]

    for d in tqdm(data):
        pred, ref = d["predicted_sparql"], d["ground_truth_sparql"]
        pred = process_output(process_output(pred,args),args)
        print("R: ",ref)
        print("P: ",pred)
        adj_pred, adj_ref = adjust(pred), adjust(ref)
        if pred.strip() == ref.strip():
            em += 1
        if adj_pred.strip() == adj_ref.strip():
            em_adj += 1

        for metric in metrics:
            checked=False
            name = metric.name()
            if name.startswith("SP"):
                metric.update((adj_pred, adj_ref))
            else:
                metric.update((pred, ref))

    pre,rec,sp_pre,sp_rec = 0.0,0.0,0.0,0.0
    for metric in metrics:
        name = metric.name()
        score = metric.compute()
        if name=="UnigramRecall":
            rec=score
        elif name=="UnigramPrecision":
            pre=score
        elif name=="SP-UnigramRecall":
            sp_rec=score
        elif name=="SP-UnigramPrecision":
            sp_pre=score
        print(name, str(score))
    f1 =  (2*pre*rec)/(pre+rec)
    sp_f1 = (2*sp_pre*sp_rec)/(sp_pre+sp_rec)
    print("F1-score: ", round(f1,4))
    print("SP-F1-score: ", round(sp_f1,4))
    print("Exact Match ", round(em/len(data) , 4))
    print("Exact Match adjusted ", round(em_adj / len(data), 4))
    print(f"Exactly matched {em}/{len(data)} times {round(em/len(data) , 4)}%")
