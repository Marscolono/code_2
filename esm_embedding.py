import os
import torch
import pandas as pd
import pathlib
from esm import pretrained, FastaBatchedDataset

class hyperParameter:
    def __init__(self) -> None:
        # self.input_data_dir = "/lizutan/code/MCANet/dataset/pipeline_5/identity_0.5_shuffle_1"
        self.input_data_dir = "/lizutan/code/MCANet/dataset/pipeline_10/rec_0.8_shuffle_1"
        self.toks_per_batch = 700 * 128
        self.truncation_seq_length = 700
        self.repr_layers = [33]
        self.include = ['per_tok']
        self.output_dir = pathlib.Path("/lizutan/code/MCANet/preprocess/esm_embedding_data/protein_700_pipeline_10")
        self.nogpu = False

#args = hyperParameter()

def main(args):
    # Load ESM-2 model
    model, alphabet = pretrained.load_model_and_alphabet("/lizutan/code/MCANet/preprocess/esm2_t33_650M_UR50D.pt")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    model.eval()  # disables dropout for deterministic results

    # Prepare data 
    rec_index, rec_str = [], []
    for input_file in os.listdir(args.input_data_dir):
        # default_columns = ['attB','attP', 'rec', 'attP_str', 'attP_str', 'rec_str', "label"]
        # input_table = pd.read_table("{}/{}".format(args.input_data_dir, input_file), header=None)
        input_table = pd.read_table("{}/{}".format(args.input_data_dir, input_file))#, header=True)
        # input_table.columns = default_columns
        for index in input_table.index:
            this_rec_index = input_table.loc[index, 'rec']
            this_rec_str = input_table.loc[index, 'rec_str']
            this_rec_str = this_rec_str[:args.truncation_seq_length] + "<pad>" * max(args.truncation_seq_length - len(this_rec_str), 0)
            this_rec_str = this_rec_str.replace("*", "<pad>")
            if not this_rec_index in rec_index:
                rec_index.append(this_rec_index)
                rec_str.append(this_rec_str)
            print("\rrec_index: {}".format(len(rec_index)), end="")

    dataset = FastaBatchedDataset(rec_index, rec_str)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(args.truncation_seq_length), batch_sampler=batches
    )

    os.makedirs(args.output_dir, exist_ok=True)
    return_contacts = "contacts" in args.include

    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in args.repr_layers]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and not args.nogpu:
                toks = toks.to(device="cuda", non_blocking=True)
            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")
            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                if "mean" in args.include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in args.include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[i, : truncate_len, : truncate_len].clone()

                torch.save(result, args.output_file)


if __name__ == "__main__":
    args = hyperParameter()
    main(args)