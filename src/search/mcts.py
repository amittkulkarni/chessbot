import math
import time
import chess
import numpy as np
import onnxruntime as ort

BATCH_SIZE = 8

class ONNXEngine:
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print(f"Inference Engine Loaded on: {self.session.get_providers()[0]}")
        self.input_name = self.session.get_inputs()[0].name

    def predict_batch(self, fens):
        batch_tensors = []
        for fen in fens:
            board = chess.Board(fen)
            batch_tensors.append(self._board_to_numpy(board))

        if not batch_tensors:
            return [], []

        full_batch = np.concatenate(batch_tensors, axis=0)
        outs = self.session.run(None, {self.input_name: full_batch})

        policy_logits_batch, value_logits_batch = outs[0], outs[1]
        results_policy, results_value = [], []

        for i in range(len(fens)):
            p_logits = policy_logits_batch[i]
            exp_p = np.exp(p_logits - np.max(p_logits))
            results_policy.append(exp_p / exp_p.sum())

            v_logits = value_logits_batch[i]
            exp_v = np.exp(v_logits - np.max(v_logits))
            v_probs = exp_v / exp_v.sum()

            # WDL Score: Win + 0.5 * Draw
            score = v_probs[0] + (0.5 * v_probs[1])
            if " w " not in fens[i]:
                score = 1.0 - score
            results_value.append(score)

        return results_policy, results_value

    def _board_to_numpy(self, board: chess.Board):
        PIECE_MAP = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5, 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
        matrix = np.zeros((18, 8, 8), dtype=np.float32)
        for sq, piece in board.piece_map().items():
            row, col = divmod(sq, 8)
            matrix[PIECE_MAP[piece.symbol()], 7 - row, col] = 1.0
        if board.has_kingside_castling_rights(chess.WHITE): matrix[12, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE): matrix[13, :, :] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK): matrix[14, :, :] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK): matrix[15, :, :] = 1.0
        if board.turn == chess.WHITE: matrix[16, :, :] = 1.0
        matrix[17, :, :] = board.halfmove_clock / 100.0
        return np.expand_dims(matrix, axis=0)


class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0
        self.is_expanded = False

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0


class BatchMCTS:
    def __init__(self, engine: ONNXEngine, c_puct: float = 0.9):
        self.engine = engine
        self.c_puct = c_puct
        self.root = None

    def step_root(self, move):
        if self.root is not None and move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = None

    def run(self, fen: str, time_limit: float = 5.0):
        if self.root is None or self.root.game_state.fen() != fen:
            self.root = MCTSNode(chess.Board(fen))
            pol, val = self.engine.predict_batch([fen])
            self._expand(self.root, pol[0])
            self.root.value_sum += val[0]
            self.root.visit_count += 1

        start_time = time.time()
        simulations = 0

        while True:
            if simulations % 20 == 0 and (time.time() - start_time) > time_limit:
                break

            leaf_nodes = []

            # --- SELECTION ---
            for _ in range(BATCH_SIZE):
                node = self.root
                while node.is_expanded and len(node.children) > 0:
                    node = self._select_child(node)
                node.visit_count += 1
                leaf_nodes.append(node)

            # --- EVALUATION ---
            valid_leafs, valid_fens = [], []
            for node in leaf_nodes:
                node.visit_count -= 1
                outcome = node.game_state.outcome()
                if outcome:
                    val = 1.0 if outcome.winner == chess.WHITE else 0.0 if outcome.winner == chess.BLACK else 0.5
                    self._backpropagate(node, val)
                else:
                    valid_leafs.append(node)
                    valid_fens.append(node.game_state.fen())

            if not valid_fens: continue

            # --- BATCH INFERENCE ---
            policies, values = self.engine.predict_batch(valid_fens)

            # --- EXPANSION & BACKPROP ---
            for i, node in enumerate(valid_leafs):
                self._expand(node, policies[i])
                self._backpropagate(node, values[i])
            simulations += BATCH_SIZE

        if not self.root.children:
            return list(self.root.game_state.legal_moves)[0], "Mate"

        best_child = max(self.root.children.values(), key=lambda n: n.visit_count)

        wr = best_child.value if self.root.game_state.turn == chess.WHITE else 1.0 - best_child.value
        score_str = f"{(-400 * math.log(1/wr - 1))/100.0:+.2f}" if 0.001 < wr < 0.999 else ("+M" if wr > 0.5 else "-M")

        print(f"⚡ Sim: {simulations} | Depth: {self._get_depth(best_child)} | Score: {score_str}")
        return best_child.move, score_str

    def _select_child(self, node):
        best_score = -float('inf')
        best_child = None
        for child in node.children.values():
            q = child.value if node.game_state.turn == chess.WHITE else 1.0 - child.value
            u = self.c_puct * child.prior * (math.sqrt(node.visit_count) / (1 + child.visit_count))
            if (q + u) > best_score:
                best_score = q + u
                best_child = child
        return best_child

    def _expand(self, node, policy):
        node.is_expanded = True
        sum_probs = 0
        for move in node.game_state.legal_moves:
            idx = (move.from_square * 64) + move.to_square
            if idx < 4096:
                prob = policy[idx]
                sum_probs += prob
                next_state = node.game_state.copy(stack=False)
                next_state.push(move)
                node.children[move] = MCTSNode(next_state, parent=node, move=move, prior=prob)
        if sum_probs > 0:
            for child in node.children.values():
                child.prior /= sum_probs

    def _backpropagate(self, node, value):
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _get_depth(self, node):
        d = 0
        while node:
            d += 1
            if not node.children: break
            node = max(node.children.values(), key=lambda n: n.visit_count)
        return d