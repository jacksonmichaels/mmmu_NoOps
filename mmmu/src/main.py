from utils import parse_arguments, set_device_and_seed, load_config
from config import MODEL_DICT_LLMs, all_subs
from dataset import prepare_NoOp, prepare_dataset
from model_loader import load_model_and_tokenizer
from inference import run_inference, save_results

def main():
    args = parse_arguments()
    subjects = all_subs if args.subject == "all" else [args.subject]
    device = set_device_and_seed(args.seed, args.device)
    args.config = load_config(args.config_path)

    if args.noop != "none":
        dataset = prepare_NoOp(args.data_path, args.split, subjects, args.noop)
    else:
        dataset = prepare_dataset(args.data_path, args.split, subjects)

    model, tokenizer, processor = load_model_and_tokenizer(args, device, MODEL_DICT_LLMs)
    eval_only_results, parse_and_eval_results = run_inference(dataset, model, processor, args, tokenizer)
    save_results("results/" + args.output_path, eval_only_results, parse_and_eval_results)

if __name__ == "__main__":
    main()
