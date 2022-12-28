import spacy
import json
from tqdm import tqdm

nlp = spacy.load("en_core_web_lg")
for label in nlp.get_pipe("tagger").labels:
    print(label, " -- ", spacy.explain(label))
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
text = "What's the name of the president of united states of America and embedding ?"
doc = nlp(text)
for token in doc:
  print(token, token.pos_)

tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
print(tokenized_ids)


def align_tokens( ques, tokenized_ques):
    speical = "Ġ"
    tok = [s.replace(speical, "") for s in tokenized_ques]
    doc = nlp(ques)

    t = list()
    tp = list()
    ps = list()
    childlist = list()
    treelevel = {i: -1 for i in range(len(tok))}
    root = ""
    current_child = list()
    for token in doc:
        # print(token.text, '  -> ',token.dep_)
        t.append(token.text)
        tp.append(token.dep_)
        ps.append(token.pos_)
        childlist.append([child for child in token.children])

    allchilds = [child for child in childlist if child]
    listdone = [-1 for alist in allchilds]

    kk = 0
    while (kk < len(t)):
        for lvl, adep in enumerate(allchilds):
            if listdone[lvl] == -1:
                for tlst, ac in enumerate(adep):
                    for k, v in treelevel.items():
                        if treelevel[k] == -1 and t[k] == ac.text:
                            treelevel[k] = lvl + 2
                            break
                listdone[lvl] = 1
        kk += 1

    # root check
    temp = treelevel.copy()
    for m, n in temp.items():
        if n == -1:
            treelevel[m] = 1
            break

    mapping = {i: {"token": None, "pos": None, "dep-tok": None, "dep-lvl": None} for i in range(len(tok))}

    j = 0

    for i, a in enumerate(tok):
        for k, et in enumerate(t):
            if a in et and j >= k:
                mapping[i]["token"] = a
                mapping[i]["pos"] = ps[k]
                mapping[i]["dep-tok"] = tp[k]
                mapping[i]["dep-lvl"] = treelevel[k]
                j += 1
    return [b["dep-tok"] for a, b in mapping.items()], [b["dep-lvl"] for a, b in mapping.items()], mapping


pos_mapping = {}
dep_mapping = {}
currentid = 0
depids = set()
splits = ["train", "test", "val"]
types = set()
dataset = "vquanda"

for sp in splits:
    data = json.load(open(f"data/{dataset}/original/{sp}.json"))
    pos_data = list()
    for d in tqdm(data, desc=f"{sp}"):
        pos_list = []
        pos_tokens = []
        ques = d["en_ques"].replace("?","")
        q_tok = tokenizer.tokenize(ques)
        dep_token, dep_level,mapping = align_tokens(ques,q_tok)
        for dt in dep_token:
            depids.add(dt)

        for w in q_tok:
            doc = nlp(w)
            w_type = doc[0].pos_
            if w_type not in pos_mapping:
                pos_mapping[w_type] = currentid+1
                currentid+=1
            pos_tokens+=  [w_type]
            pos_list+= [pos_mapping[w_type]]
        new_data = d.copy()
        new_data["question"] = ques
        new_data["question_pos_tokens"] = pos_tokens
        new_data["question_pos_ids"] = pos_list
        new_data["question_dep_ids"] = dep_token
        new_data["question_dep_lvl"] = dep_level
        new_data["mapping"] = mapping
        pos_data.append(new_data)
        assert len(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ques)))==len(pos_list)==len(pos_tokens)==len(dep_level)

    json.dump(pos_data, open(f"data/{dataset}/{sp}.json", "w"), indent=3)



pos_mapping_reverse = {v:k for k,v in pos_mapping.items()}
json.dump(pos_mapping_reverse, open(f"data/{dataset}/pos_mapping.json", "w"), indent=3)
dep_mapping_reverse = {dtok:anid+1 for anid, dtok in enumerate(depids)}
json.dump(dep_mapping_reverse, open(f"data/{dataset}/dep_mapping.json", "w"), indent=3)