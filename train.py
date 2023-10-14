import argparse
import logging, json
import torch
from tqdm import tqdm, trange
from argparse import Namespace
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
     AdamW,
     AutoConfig,
     AutoTokenizer,
     PreTrainedModel,
     PreTrainedTokenizer,
     get_linear_schedule_with_warmup,
)

from scripts.model import run_batch_generation, GPT2LMHeadModel

import os
from utils.args import (
    set_default_params,
    update_additional_params
)
from torch.optim import lr_scheduler
import random
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn_train, run_batch_fn_eval) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        log_dir = os.path.join("runs", args.exp_name, args.dataset) if args.exp_name else None
        tb_writer = SummaryWriter(log_dir)
        args.output_dir = log_dir
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=train_dataset.collate_fn
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    global_eval = 99999
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    if args.scheduler=="linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    else:
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=0.0001, verbose=True)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    global_step = 0
    model.zero_grad()
    train_iterator = trange(0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])

    set_seed(args)  # for reproducibility

    for _ in train_iterator:
        tr_loss = 0.0
        local_steps = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            loss, _, _, _ = run_batch_fn_train(args, model, batch)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                if args.scheduler=="linear":
                    scheduler.step()
                else:
                    pass
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss / local_steps)

        results = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=str(global_step))
        if args.local_rank in [-1, 0]:
            for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)

            tb_writer.add_scalar("loss", tr_loss / local_steps, global_step)
            if args.scheduler == "reducelr":
                scheduler.step(results["loss"])
                tb_writer.add_scalar("lr", args.learning_rate)
            else:
                tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], global_step)


            checkpoint_prefix = "checkpoint"
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training

            logger.info("Saving model checkpoint to %s", output_dir)
            global_eval=results["loss"]
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            with open(os.path.join(output_dir, "params.json"), "w") as jsonfile:
                json.dump(args.params, jsonfile, indent=2, default=lambda x: str(x))
                logger.info("Saving model checkpoint to %s", output_dir)

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step

def evaluate(args, eval_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, run_batch_fn, desc="") -> Dict:
    if args.local_rank in [-1, 0]:
        eval_output_dir = args.output_dir
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=eval_dataset.collate_fn
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and (eval_dataset.args.eval_all_snippets):
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    data_infos = []
    all_preds = []
    all_labels = []
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
        with torch.no_grad():
            loss, lm_logits, mc_logits, mc_labels = run_batch_fn(args, model, batch)
            all_preds.append(mc_logits.detach().cpu().numpy())
            all_labels.append(mc_labels.detach().cpu().numpy())
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps

    if args.task.lower() == "generation":
        perplexity = torch.exp(torch.tensor(eval_loss))
        result = {"perplexity": perplexity, "loss": eval_loss}

    if args.local_rank in [-1, 0]:
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as writer:
            logger.info("***** Eval results %s *****" % desc)
            writer.write("***** Eval results %s *****\n" % desc)
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--params_file", type=str, default="config/gpt-2-base/params.json", help="JSON configuration file")
    parser.add_argument("--eval_only", action="store_true",
                        help="Perform evaluation only")
    parser.add_argument("--checkpoint", type=str, help="Saved checkpoint directory")
    parser.add_argument("--history_max_tokens", type=int, default=-1,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--epochs", type=int, default=-1, help="number of epochs for training")
    parser.add_argument("--knowledge_max_tokens", type=int, default=-1, help="Maximum length in tokens for knowledge, will override that value in config.")
    parser.add_argument("--dataroot", type=str, default="data", help="Path to dataset.")
    parser.add_argument("--dataset", type=str, default="lcquad2", choices=["lcquad2","qald9", "vquanda"],
                        help="dataset name.")
    parser.add_argument('--eval_partial', action='store_true')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--knowledge', action='store_true')

    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear","reducelr"])
    parser.add_argument("--eval_dataset", type=str, default="val",
                        help="Dataset to evaluate on, will load dataset from {dataroot}/{eval_dataset}")
    parser.add_argument("--output_file", type=str, default="", help="Predictions will be written to this file.")
    parser.add_argument("--negative_sample_method", type=str, choices=["all", "mix", "oracle"],
                        default="all",
                        help="Negative sampling method for knowledge selection, will override the value in config.")
    parser.add_argument("--eval_all_snippets", action='store_true',
                        help="If set, the candidates to be selected would be all knowledge snippets, not sampled subset.")
    parser.add_argument("--exp_name", type=str, default="sgpt",
                        help="Name of the experiment, checkpoints will be stored in runs/{exp_name}")
    parser.add_argument("--eval_desc", type=str, default="",
                        help="Optional description to be listed in eval_results.txt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d : %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    if args.dataset=="lcquad2":
        from scripts.dataset_lcquad2 import EvalDataset, Dataset, SPECIAL_TOKENS
    if args.dataset=="vquanda":
        from scripts.dataset_vquanda import EvalDataset, Dataset, SPECIAL_TOKENS
    if args.dataset=="qald9":
        from scripts.dataset_qald9 import EvalDataset, Dataset, SPECIAL_TOKENS


    args = parser.parse_args()
    fromcommand = args
    params = json.load(open(args.params_file, "r"))
    params["num_train_epochs"]= fromcommand.epochs
    args = vars(args)
    update_additional_params(params, args)
    args.update(params)
    args = Namespace(**args)


    args.params = params  # used for saving checkpoints
    if fromcommand.epochs!=-1:
        args.num_train_epochs = fromcommand.epochs


    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    set_default_params(args)
    dataset_args = Namespace(**args.dataset_args)
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

    set_seed(args)
    args.train_batch_size = 2

    dataset_class, model_class, run_batch_fn_train, run_batch_fn_eval = Dataset, GPT2LMHeadModel, run_batch_generation, run_batch_generation

    if args.eval_only:
        pass
    else:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        # set output_past to False for DataParallel to work during evaluation
        config.output_past = False
        config.knowledge_max_tokens = dataset_args.knowledge_max_tokens
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.resize_token_embeddings(len(tokenizer))

    model.to(args.device)

    #####   training    #######
    if not args.eval_only:

        train_dataset = dataset_class(dataset_args, tokenizer, name=args.dataset, split_type="train", masked=args.masked ,eval_partial=None)
        eval_dataset = dataset_class(dataset_args, tokenizer, name=args.dataset,split_type="val", masked=args.masked, eval_partial=None)

        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      collate_fn=train_dataset.collate_fn)

        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        global_step = train(args, train_dataset, eval_dataset, model, tokenizer, run_batch_fn_train, run_batch_fn_eval)
        #logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
        if args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir, exist_ok=True)

            logger.info("Saving model checkpoint to %s", args.output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            with open(os.path.join(args.output_dir, "params.json"), "w") as jsonfile:
                json.dump(params, jsonfile, indent=2)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.output_dir,)
            model.to(args.device)

    # Evaluation
    result = {}
    if args.local_rank in [-1, 0]:
        eval_dataset = dataset_class(dataset_args, tokenizer, name=args.dataset, split_type=args.eval_dataset, masked=args.masked, eval_partial=None)
        result = evaluate(args, eval_dataset, model, tokenizer, run_batch_fn_eval, desc=args.eval_desc or args.eval_dataset)
    return result


if __name__=="__main__":
    main()