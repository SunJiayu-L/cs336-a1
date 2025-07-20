import regex as re
import collections
import os
from typing import BinaryIO

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def bpe_train(
    input_path, 
    vocab_size: int, 
    special_tokens: list[str], 
    num_processes: int = 4
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a BPE tokenizer on the given input corpus.
    
    Args:
        input_path: Path to the training corpus file
        vocab_size: Total vocabulary size including special tokens
        special_tokens: List of special tokens to add to vocabulary
        num_processes: Number of processes for parallel processing
    
    Returns:
        tuple of (vocab, merges) where:
        - vocab: Dictionary mapping token IDs to byte sequences
        - merges: List of merge operations in order of application
    """
    # Calculate number of merges needed
    num_of_merge = vocab_size - 256 - len(special_tokens)
    if num_of_merge <= 0:
        raise ValueError(
            f"vocab_size {vocab_size} is too small. "
            f"Needs to be at least {256 + len(special_tokens)}"
        )
    
    # 1. initialize the vocab
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for token in special_tokens:
        vocab[next_id] = token.encode('utf-8')
        next_id += 1

    # Track merges for return value
    merges = []

    # 2. parallel processing of pretoken
    table: dict[tuple[bytes], int] = collections.defaultdict(int)
    with open(input_path, 'rb') as f:
        # Find chunk boundaries
        boundaries = find_chunk_boundaries(
            f, num_processes, "<|endoftext|>".encode("utf-8"))
        
        PAT = (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| """
               r"""?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        # escape special tokens
        escaped_special_tokens = [re.escape(token) for token in special_tokens]
        # split pattern for tokenization
        split_pattern = '|'.join(escaped_special_tokens + [PAT])
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            # pre-tokenize the chunk
            text_parts = re.findall(split_pattern, chunk)
            text_parts = [part for part in text_parts if part]
            
            # ！！！！ accumulate token counts from this chunk
            for token in text_parts:
                if token in special_tokens:
                    # Special tokens should be treated as single units
                    byte_sequence = (token.encode('utf-8'),)
                else:
                    # Regular tokens: each byte becomes a separate element
                    token_bytes = token.encode('utf-8')
                    byte_sequence = tuple(bytes([b]) for b in token_bytes)
                table[byte_sequence] += 1

    # 3. find and merge the most frequent pairs
    for _ in range(num_of_merge):
        pairs = collections.defaultdict(int)
        # 找到所有相邻的字节对
        for byte_sequence, freq in table.items():
            for i in range(len(byte_sequence) - 1):
                pair = (byte_sequence[i], byte_sequence[i + 1])
                pairs[pair] += freq
        
        if not pairs:  # 如果没有更多的对可以合并，则停止
            break
            
        #！！！！ 找到频率最高的字节对，在频率相同时按字典序排序以确保确定性
        max_pair = max(pairs, key=lambda pair: (pairs[pair], pair))
        
        # Record this merge
        merges.append(max_pair)
        
        # 在table中合并这个字节对
        new_table = collections.defaultdict(int)
        for byte_sequence, freq in table.items():
            new_sequence = []
            i = 0
            while i < len(byte_sequence):
                # 如果找到了要合并的对
                if (i < len(byte_sequence) - 1 and 
                    (byte_sequence[i], byte_sequence[i + 1]) == max_pair):
                    # 合并这两个字节
                    merged_bytes = byte_sequence[i] + byte_sequence[i + 1]
                    new_sequence.append(merged_bytes)
                    i += 2  # 跳过下一个字节，因为已经合并了
                else:
                    new_sequence.append(byte_sequence[i])
                    i += 1
            new_table[tuple(new_sequence)] = freq
        
        table = new_table
        
        # 将合并后的字节对添加到词汇表中
        merged_token = max_pair[0] + max_pair[1]
        vocab[next_id] = merged_token
        next_id += 1

    return vocab, merges

