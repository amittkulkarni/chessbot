import os
import glob
import numpy as np
import torch
import chess
from torch.utils.data import IterableDataset

PIECE_MAP = {'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5, 'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11}

def encode_move(move_uci: str) -> int:
    try:
        fr = chess.parse_square(move_uci[:2])
        to = chess.parse_square(move_uci[2:4])
        return (fr * 64) + to
    except:
        return 0

def fen_to_tensor_18ch(fen: str) -> np.ndarray:
    parts = fen.split(' ')
    board_str, turn, castling = parts[0], parts[1], parts[2]
    try: halfmove = parts[4]
    except: halfmove = 0

    matrix = np.zeros((18, 8, 8), dtype=np.float32)
    rows = board_str.split('/')
    for row_idx, row_data in enumerate(rows):
        col_idx = 0
        for char in row_data:
            if char.isdigit():
                col_idx += int(char)
            else:
                matrix[PIECE_MAP[char], row_idx, col_idx] = 1.0
                col_idx += 1

    if 'K' in castling: matrix[12, :, :] = 1.0
    if 'Q' in castling: matrix[13, :, :] = 1.0
    if 'k' in castling: matrix[14, :, :] = 1.0
    if 'q' in castling: matrix[15, :, :] = 1.0
    if turn == 'w': matrix[16, :, :] = 1.0
    try: matrix[17, :, :] = float(halfmove) / 100.0
    except: pass

    return matrix

class ChessDataset18(IterableDataset):
    def __init__(self, root_dir: str):
        self.files = []
        data_folders = glob.glob(os.path.join(root_dir, "data_*"))

        for folder in data_folders:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if not file.startswith('.'):
                        self.files.append(os.path.join(root, file))

        print(f"✅ Total Training Files Found: {len(self.files)}")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        np.random.shuffle(self.files)

        if worker_info:
            per_worker = int(np.ceil(len(self.files) / float(worker_info.num_workers)))
            start = worker_info.id * per_worker
            end = min(start + per_worker, len(self.files))
            my_files = self.files[start:end]
        else:
            my_files = self.files

        for f_path in my_files:
            try:
                with open(f_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split('|')
                        if len(parts) < 4: continue
                        fen, move_uci, score_str = parts[0], parts[1], parts[3]

                        tensor = fen_to_tensor_18ch(fen)
                        move_id = encode_move(move_uci)

                        try: score_val = max(0.0, min(1.0, float(score_str)))
                        except: score_val = 0.5

                        yield tensor, move_id, score_val
            except Exception:
                pass