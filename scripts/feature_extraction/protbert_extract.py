import torch
from sklearn.decomposition import KernelPCA
from transformers import BertModel, BertTokenizer

from scripts.utils.read_data import read_data_sequences
from scripts.utils.write_data import write_list


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_components = 30


def main():
    """
    For each protein sequence, feature extraction of the embeddings computed
    by ProtBert
    """
    sequences = read_data_sequences()
    sequences = [" ".join(seq) for seq in sequences]

    tokenizer_bert = "Rostlab/prot_bert"
    tokenizer = BertTokenizer.from_pretrained(tokenizer_bert, do_lower_case=False)

    model_bert = "Rostlab/prot_bert"
    model_fe = BertModel.from_pretrained(model_bert).to(device)

    train_embeddings_list_kpca = []
    for i, protein in enumerate(sequences):
        kpca = KernelPCA(n_components=n_components)
        encoded_input = tokenizer(
            protein,
            add_special_tokens=True,
            padding=True,
            is_split_into_words=True,
            return_tensors="pt",
        ).to(device)
        output = model_fe(**encoded_input).last_hidden_state[0].detach().cpu().numpy()
        output = kpca.fit_transform(output)
        train_embeddings_list_kpca.append(output)
        if i % 500 == 0:
            print(i)

    write_list(train_embeddings_list_kpca, "embeddings_sequence")


if __name__ == "__main__":
    main()
