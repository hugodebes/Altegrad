import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot


class ProteinSeqdataset(Dataset):
    """
    Customised Dataset : from Proteins to Tensor
    """

    def __init__(self, proteins_seq, y=None, vocab=None, padding_mark="PAD"):
        """
        Build the tensor of samples

        Arguments
        ---------
            proetins_seq: list(str)
                List of proetins each composed of Amino Acids
            y: list(str)
                List of Labels (can be None if building the test dataset)
            vocab: (aa2id,id2aa)
                Map the Amino Acid to an index and vice-versa
            padding_mark: str
                Symbol used to denote the padding
        """

        self.y = y
        self.padding_mark = padding_mark
        self.proteins_seq = proteins_seq
        # Maximum length of sequences
        self.max_len_sequence = max(
            [len(proteins_seq[i]) for i in range(len(proteins_seq))]
        )

        # length of each sequence
        self.lengths = torch.Tensor([len(prot) for prot in self.proteins_seq])

        # Allow to import a vocabulary (for valid/test datasets, that will use the training vocabulary)
        if vocab is not None:
            self.aa2id, self.id2aa = vocab
        else:
            # If no vocabulary imported, build it (and reverse)
            self.aa2id, self.id2aa = self.build_vocab()
        self.vocab_size = len(self.aa2id)

        # Convert to Tensor and apply the vocabulary
        sequences_tensor = list(
            map(
                lambda prot: torch.Tensor([self.aa2id[aa] for aa in prot]),
                self.proteins_seq,
            )
        )
        # Pad the sequence
        sequences_padded = pad_sequence(
            sequences_tensor, batch_first=True, padding_value=0
        )

        # One Hot encoding
        sequences_encoded = one_hot(
            sequences_padded.to(torch.int64), num_classes=self.vocab_size
        )

        # Convert to Torch
        self.X = sequences_encoded
        self.y = torch.Tensor(self.y)

    def get_data(self):
        return self.X, self.y

    def get_lengths(self):
        return self.lengths

    def __len__(self):
        return len(self.proteins_seq)

    def __getitem__(self, idx):
        # The iterator just gets one particular example with its category
        # The dataloader will take care of the shuffling and batching
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (
            self.X[idx],
            self.y[idx],
            self.lengths[idx],
        )  # Add lengths for padding purpose

    def build_vocab(self):
        """
        Function to build the Amino Acids vocabulary

        Returns
        ----------------------------
          aa2id : <dict{str:int}>
              Dictionary to go from an Amino Acids to an id
          id2aa : <dict{int:str}>
              Dictionary to go from an id to an Amino Acids
        """
        # Set of the Amino Acids
        amino_acids = list(set("".join(self.proteins_seq)))
        # print(amino_acids)
        # Build Vocab (+1 for the padding)
        aa2id = {aa: i + 1 for i, aa in enumerate(amino_acids)}
        id2aa = {i + 1: aa for i, aa in enumerate(amino_acids)}

        # Add the Padding id
        aa2id = {"PAD": self.padding_mark, **aa2id}
        id2aa = {**id2aa, self.padding_mark: "PAD"}

        return aa2id, id2aa

    def get_vocab(self):
        # A simple way to get the training vocab when building the valid/test
        return self.aa2id, self.id2aa
