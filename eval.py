import argparse
import glob
import logging
import os
import random
import shutil
import json

from typing import Dict
from argparse import Namespace
import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from scripts.model import GPT2LMHeadModel

from utils.args import update_additional_params
from scripts.model import decode_sample
from utils.metrics import (
    UnigramMetric, NGramDiversity,
    CorpusNGramDiversity,
    BLEU, METEOR, ROUGE, SPBLEU, SPUnigramMetric
)
from utils.data import write_generation_preds

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

def is_num(text):
    """ Check if the string is a number"""
    text = text.replace("\'","")
    text = text.replace('\"', "")
    return text.replace('.','',1).isdigit()


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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def normalized_vars(query):
    ref_vars = list()
    cleaned_query = query
    for token in query.split():
        if token.startswith("?") and token not in ref_vars:
            ref_vars.append(token)

    for i,var in enumerate(ref_vars):
        cleaned_query = cleaned_query.replace(var,f"var{i+1}")
    return cleaned_query


def evaluate(args, eval_dataset, model, tokenizer, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=1,  # only support batch_size=1 for sampling right now
        collate_fn=eval_dataset.collate_fn
    )

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

    args.tokenizer = tokenizer
    all_output_texts = []
    all_ground_truths = []
    dialog_ids = []
    tasks = []
    do_evaluate = False
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            sampled_output_ids, ground_truth, query_id, knowledge_text = decode_sample(args, model, batch, eval_dataset)
            sampled_output_text = tokenizer.decode(sampled_output_ids, skip_special_tokens=True)
            print("Truth: ", ground_truth)
            sampled_output_text = process_output(process_output(sampled_output_text, args), args)
            print("Predt: ", sampled_output_text)
            all_output_texts.append(sampled_output_text)
            all_ground_truths.append(ground_truth)
            dialog_ids.append(query_id)
            sampled_output_text_norm, ground_truth_norm = normalized_vars(sampled_output_text), normalized_vars(ground_truth)
        if ground_truth.strip() != "":
            do_evaluate = True
            for metric in metrics:
                metric.update((sampled_output_text, ground_truth))
                name = metric.name()
                if name.startswith("SP"):
                    metric.update((sampled_output_text_norm, ground_truth_norm))
                else:
                    metric.update((sampled_output_text, ground_truth))

    if args.output_file:
        write_generation_preds(args.output_file, dialog_ids, all_output_texts, all_ground_truths)

    result = dict()
    if do_evaluate and args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        pre, rec, sp_pre, sp_rec = 0.0, 0.0, 0.0, 0.0
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for metric in metrics:
                name = metric.name()
                score = metric.compute()
                if name == "UnigramRecall":
                    rec = score
                elif name == "UnigramPrecision":
                    pre = score
                elif name == "SP-UnigramRecall":
                    sp_rec = score
                elif name == "SP-UnigramPrecision":
                    sp_pre = score
                print(name, str(score))
                result[name] = score
                logger.info("  %s = %s", name, str(score))
                writer.write("%s = %s\n" % (name, str(score)))
            f1 = (2 * pre * rec) / (pre + rec)
            sp_f1 = (2 * sp_pre * sp_rec) / (sp_pre + sp_rec)
            print("F1-score: ", round(f1, 4))
            print("SP-F1-score: ", round(sp_f1, 4))

    return result


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--eval_partial', action='store_true')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument("--decode", type=str, default="basic", choices=["basic","beam"], help="decoding technique")
    parser.add_argument('--dataset', type=str, default='lcquad2', choices=['lcquad2','qald9','vquanda'])
    parser.add_argument('--knowledge', action='store_true')
    parser.add_argument("--generation_params_file", type=str, default="config/gpt-2-base/generation_params.json",
                        help="JSON configuration file for generation-related configurations.")
    parser.add_argument("--dataroot", type=str, default="data",
                        help="Path to dataset, will override the path in config.")
    parser.add_argument("--eval_dataset", type=str, default="test",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    if args.dataset=="lcquad2":
        from scripts.dataset_lcquad2 import EvalDataset
    if args.dataset == "vquanda":
        from scripts.dataset_vquanda import EvalDataset
    if args.dataset=="qald9":
        from scripts.dataset_qald9 import EvalDataset, Dataset, SPECIAL_TOKENS

    # load args from params file and update the args Namespace
    args.params_file = os.path.join(args.checkpoint, "params.json")
    with open(args.params_file, "r") as f:
        params = json.load(f)
        args = vars(args)
        update_additional_params(params, args)
        args.update(params)
        if len(args["generation_params_file"]) > 0:
            with open(args["generation_params_file"]) as fg:
                generation_params = json.load(fg)
            args.update(generation_params)
        args = Namespace(**args)

    args.params = params  # used for saving checkpoints
    dataset_args = Namespace(**args.dataset_args)
    dataset_args.local_rank = args.local_rank
    dataset_args.task = args.task
    dataset_args.knowledge = args.knowledge

    # Setup CUDA, GPU & distributed training
    args.distributed = (args.local_rank != -1)
    if not args.distributed:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
    args.n_gpu = torch.cuda.device_count() if not args.distributed else 1
    args.device = device

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.output_dir = args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Generation parameters %s", args)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = EvalDataset(dataset_args, tokenizer, name=args.dataset, split_type=args.eval_dataset, masked=args.masked, eval_partial=args.eval_partial)
        result = evaluate(args, eval_dataset, model, tokenizer, desc=args.eval_desc or args.eval_dataset)
    return result


if __name__ == "__main__":
    main()
