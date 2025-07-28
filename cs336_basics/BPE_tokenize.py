class BPE_tokenize:

    def __init__(self, vocab_file, merge, special_tokens=None):
        """
        Initializes the BPE tokenizer.

        Args:
            vocab_file (str): Path to the vocabulary file. dict[int,bytes]
            merge (str): Path to the merge file.   list[tuple[bytes, bytes]]
            special_tokens (list, optional): List of special tokens to add to the tokenizer. list[str]
        """
        self.vocab_file = vocab_file
        self.merge = merge
        self.special_tokens = special_tokens or []

    @classmethod
    def from_file(cls, vocab_file, merge, special_tokens=None):
        """
        Creates a BPE tokenizer from the specified vocabulary and merge files.

        Args:
            vocab_file (str): Path to the vocabulary file.
            merge (str): Path to the merge file.
            special_tokens (list, optional): List of special tokens to add to the tokenizer.

        Returns:
            BPE_tokenize: An instance of the BPE_tokenize class.
        """
    
    def encode(self, text:str)->list[int]:
        """
        Encodes the input text into BPE tokens.

        Args:
            text (str): The input text to encode.

        Returns:
            list: A list of BPE tokens.
        """



    def decode(self, ids:list[int])->str:
        """
        Decodes a list of BPE tokens back into text.

        Args:
            tokens (list): A list of BPE tokens to decode.

        Returns:
            str: The decoded text.
        """

    def encode_iterable(self, iterable:Iterable[str])->list[list[int]]:
        """
        Encodes an iterable of strings into BPE tokens.

        Args:
            iterable (iterable): An iterable of strings to encode.

        Returns:
            list: A list of lists, where each inner list contains BPE tokens for each string.
        """