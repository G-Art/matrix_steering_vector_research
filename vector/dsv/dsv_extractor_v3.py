import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from typing import Any, Dict, List, Tuple
from pydantic import BaseModel
from tqdm.auto import tqdm

from dsv.steering_hook_v3 import MatrixSteeringHook, register_hooks
from dsv.steereng_vector_v3 import MatrixSteeringVector


class ExtractionHook:
    """
    Hook for capturing internal activations during generation.
    Stores data on CPU to save VRAM.

    Хук для захоплення внутрішніх активацій під час генерації.
    Зберігає дані на CPU для економії відеопам'яті.
    """

    def __init__(self, layer_idx: int, storage: List[torch.Tensor], is_prompt_only: bool = False):
        self.layer_idx = layer_idx
        self.storage = storage
        self.is_prompt_only = is_prompt_only

    def __call__(self, module, args):
        x = args[0]  # [Batch, Seq, Hidden]
        # Detach and move to CPU immediately / Від'єднуємо та переносимо на CPU одразу
        data = x[0].detach().cpu()

        if self.is_prompt_only:
            self.storage.append(data[-1:].clone())
        else:
            self.storage.append(data.clone())
        return None


# ==========================================
# 2. EXTRACTOR / ЕКСТРАКТОР
# ==========================================

class ActivationExtractorConfig(BaseModel):
    do_sample: bool = False


class ActivationExtractor:
    """
    Helper class to run model generation and collect traces.

    Допоміжний клас для запуску генерації моделі та збору трейсів.
    """

    def __init__(self, model, tokenizer, config: ActivationExtractorConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = model.device

    def extract_full_trace(self, text: str, num_tokens: int) -> Dict[int, torch.Tensor]:
        """
        Runs generation and returns full traces for all layers.

        Запускає генерацію та повертає повні трейси для всіх шарів.
        Returns:
            Dict[layer_idx, Tensor[Seq_Len, Hidden_Size]]
        """
        self.model.eval()
        traces = {i: [] for i in range(self.model.config.num_hidden_layers)}
        hooks = []

        # Register extraction hooks on all layers
        # Реєструємо хуки видобутку на всіх шарах
        for i in range(self.model.config.num_hidden_layers):
            h = ExtractionHook(i, traces[i], is_prompt_only=False)
            hooks.append(self.model.model.layers[i].mlp.down_proj.register_forward_pre_hook(h))

        enc = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad(), register_hooks(hooks):
            self.model.generate(
                **enc,
                max_new_tokens=num_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=self.config.do_sample
            )

        # Concatenate list of tensors into single tensor per layer
        # Об'єднуємо список тензорів в один тензор для кожного шару
        final_traces = {}
        for i, data_list in traces.items():
            if data_list:
                final_traces[i] = torch.cat(data_list, dim=0)

        return final_traces


# ==========================================
# 3. MATRIX TRAINER / МАТРИЧНИЙ ТРЕНЕР
# ==========================================

class SteeringTrainerConfig(BaseModel):
    # Tuple: (Target_Item, Source_Item) aka (French/Cheese, English/Rock)
    dataset: List[Tuple[Dict[str, Any], Dict[str, Any]]]


class MatrixIterativeTrainer:
    """
    Trainer that fits affine transformations iteratively layer by layer using Ridge Regression.

    Тренер, який навчає афінні перетворення ітеративно, шар за шаром, використовуючи гребеневу регресію.
    """

    def __init__(self, model, tokenizer, config: SteeringTrainerConfig, extractor: ActivationExtractor):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.extractor = extractor
        self.device = model.device
        self.num_layers = model.config.num_hidden_layers
        self.hidden_size = model.config.intermediate_size

        # Renamed storage for generic usage
        # Перейменовані сховища для універсального використання
        self.target_traces = {}  # ex-pure_fr (Target/Goal behavior)
        self.source_traces = {}  # ex-pure_en (Source/Base behavior)

        self.training_stats = []

    def extract_pure(self):
        """
        Phase 1: Collect Pure Baselines (Target and Source) without any steering.

        Фаза 1: Збір чистих базових ліній (Цільової та Вихідної) без жодного керування.
        """
        print(f"Phase 1: Extracting PURE traces (Source & Target)...")
        self.target_traces = {i: [] for i in range(self.num_layers)}
        self.source_traces = {i: [] for i in range(self.num_layers)}

        # item_T = Target (e.g., French prompt), item_S = Source (e.g., English prompt)
        for item_T, item_S in tqdm(self.config.dataset, desc="Pure Extraction"):
            traces_target = self.extractor.extract_full_trace(item_T["prompt"], item_T["tokens"])
            traces_source = self.extractor.extract_full_trace(item_S["prompt"], item_S["tokens"])

            for i in range(self.num_layers):
                if i in traces_target: self.target_traces[i].append(traces_target[i])
                if i in traces_source: self.source_traces[i].append(traces_source[i])

    def _capture_distorted_layer(self, layer_idx: int, active_vectors: Dict) -> List[torch.Tensor]:
        """
        Runs generation with previous vectors active to get Distorted Input X from Source prompts.

        Запускає генерацію з активними попередніми векторами для отримання спотвореного входу X з вихідних промптів.
        """
        distorted_traces = []
        hooks = []

        # Activate Previous Steering Vectors
        # Активуємо вектори керування з попередніх шарів
        for l_prev, vec_obj in active_vectors.items():
            if l_prev >= layer_idx: break
            h = MatrixSteeringHook(vec_obj, 1.0, self.device)
            hooks.append(self.model.model.layers[l_prev].mlp.down_proj.register_forward_pre_hook(h))

        # Capture Hook for current layer
        current_data = []
        h_extract = ExtractionHook(layer_idx, current_data, is_prompt_only=False)
        hooks.append(self.model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(h_extract))

        # Run generation on SOURCE prompts (item_S)
        # Запускаємо генерацію на ВИХІДНИХ промптах
        with torch.no_grad(), register_hooks(hooks):
            for _, item_S in self.config.dataset:
                current_data.clear()
                enc = self.tokenizer(item_S["prompt"], return_tensors="pt").to(self.device)
                self.model.generate(
                    **enc,
                    max_new_tokens=item_S["tokens"],
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False
                )
                if current_data:
                    distorted_traces.append(torch.cat(current_data, dim=0))
                else:
                    distorted_traces.append(torch.zeros(1, self.hidden_size))

        return distorted_traces

    def fit_ridge_matrix(self, X: np.ndarray, Y: np.ndarray, lambda_reg: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves Y = X @ W.T + B using Ridge Regression (Closed Form).

        Розв'язує рівняння Y = X @ W.T + B використовуючи Гребеневу регресію (Аналітичний розв'язок).
        """
        N, D_in = X.shape

        # 1. Calculate Centers of Mass / Знаходимо центри хмар
        mu_X = np.mean(X, axis=0, keepdims=True)  # [1, D]
        mu_Y = np.mean(Y, axis=0, keepdims=True)  # [1, D]

        # 2. Center the data / Центруємо дані
        X_c = X - mu_X
        Y_c = Y - mu_Y

        # 3. Solve for W using Ridge on CENTERED data (Pure Rotation/Scaling)
        # No bias column needed here, as data is centered!
        XTX = X_c.T @ X_c
        XTY = X_c.T @ Y_c

        # Regularization (Identity Prior or Standard)
        # Тут використовуємо Standard Ridge для чистоти обертання
        reg = lambda_reg * np.eye(D_in)

        try:
            W_T = np.linalg.solve(XTX + reg, XTY)
        except np.linalg.LinAlgError:
            W_T, _, _, _ = np.linalg.lstsq(X_c, Y_c, rcond=None)

        # W_T shape is [In, Out]

        # 4. Calculate Bias deterministically / Аналітично вираховуємо Bias
        # Bias = Target_Center - (Source_Center @ Rotation)
        B = mu_Y - (mu_X @ W_T)  # [1, Out]

        # Flatten B to 1D array
        B = B.flatten()

        return W_T.T, B  # Return W as [Out, In]

    def train_matrix(self,
                     pattern_sparse_threshold: float = 0.045,
                     impact_threshold: float = 0.05,
                     lambda_reg: float = 50.0) -> Dict[int, MatrixSteeringVector]:
        """
        Main training loop. Filters neurons and trains matrices layer by layer.

        Головний цикл тренування. Фільтрує нейрони та тренує матриці шар за шаром.
        """
        print(f"\n🚀 Starting Iterative Matrix Training (Ridge Reg={lambda_reg})...")
        trained_vectors = {}
        self.training_stats = []

        # 1. Pre-filtering Candidates based on Target vs Source difference
        # Попередня фільтрація кандидатів на основі різниці Ціль vs Джерело
        candidates = {}
        for i in range(self.num_layers):
            all_target = torch.cat(self.target_traces[i], dim=0)
            all_source = torch.cat(self.source_traces[i], dim=0)

            # Difference of means
            diff = all_target.mean(dim=0) - all_source.mean(dim=0)
            norm_diff = F.normalize(diff, dim=0, p=2)

            mask = torch.abs(norm_diff) > pattern_sparse_threshold
            indices = torch.where(mask)[0].cpu().numpy()
            if len(indices) > 0: candidates[i] = indices

        # 2. Layer Loop
        for layer_idx in range(self.num_layers):
            if layer_idx not in candidates:
                continue

            active_ids = candidates[layer_idx]

            # A. Inputs X: Distorted Source traces (what the layer sees now)
            # Входи X: Спотворені трейси Джерела (те, що шар бачить зараз)
            dist_traces = self._capture_distorted_layer(layer_idx, trained_vectors)

            # B. Targets Y: Pure Target traces (what we want the layer to output)
            # Цілі Y: Чисті трейси Цілі (те, що ми хочемо отримати на виході)
            pure_target = self.target_traces[layer_idx]

            X_list, Y_list = [], []
            for k in range(len(dist_traces)):
                x_seq = dist_traces[k]
                y_seq = pure_target[k]
                m_len = min(x_seq.shape[0], y_seq.shape[0])
                X_list.append(x_seq[:m_len, :])  # Full context input
                Y_list.append(y_seq[:m_len, active_ids])  # Active neurons output

            X_all = torch.cat(X_list, dim=0).cpu().numpy()
            Y_all = torch.cat(Y_list, dim=0).cpu().numpy()

            # C. Fit Matrix
            W, B = self.fit_ridge_matrix(X_all, Y_all, lambda_reg=lambda_reg)

            # D. Post-Filtering (Impact Check)
            # Calculate how much the matrix actually changes the values
            Y_pred = X_all @ W.T + B
            # Delta = |Predicted - Original|
            deltas = np.abs(Y_pred - X_all[:, active_ids]).mean(axis=0)

            keep_mask = deltas > impact_threshold
            final_indices = active_ids[keep_mask]

            # Logging Stats / Збір статистики
            stat = {
                "Layer": layer_idx,
                "Candidates": len(active_ids),
                "Kept": len(final_indices),
                "Pruned": len(active_ids) - len(final_indices),
                "Avg_Delta": float(deltas.mean()) if len(deltas) > 0 else 0.0,
                "Max_Delta": float(deltas.max()) if len(deltas) > 0 else 0.0
            }
            self.training_stats.append(stat)

            if len(final_indices) == 0:
                print(f"Layer {layer_idx}: Dropped all neurons (Low Impact).")
                continue

            # Slice W and B to keep only final indices
            W_final = W[keep_mask, :]
            B_final = B[keep_mask]

            print(
                f"Layer {layer_idx}: Fitted. Kept {len(final_indices)}/{len(active_ids)} neurons. (Avg Delta: {stat['Avg_Delta']:.3f})")

            # Create and store the vector
            vec = MatrixSteeringVector(
                layer_index=layer_idx,
                intermediate_size=self.hidden_size,
                active_indices=final_indices.tolist(),
                weight_matrix=W_final.tolist(),
                bias_vector=B_final.tolist()
            )
            trained_vectors[layer_idx] = vec

            # Cleanup
            del X_all, Y_all, W, B, W_final, B_final
            gc.collect()

        self.print_summary()
        return trained_vectors

    def print_summary(self):
        """
        Prints a tabular summary of the training process.
        Виводить табличне зведення процесу тренування.
        """
        print("\n" + "=" * 65)
        print(f"{'Layer':<6} | {'Candidates':<10} | {'Kept':<6} | {'Avg Delta':<10} | {'Max Delta':<10}")
        print("-" * 65)
        total_kept = 0
        for s in self.training_stats:
            print(
                f"{s['Layer']:<6} | {s['Candidates']:<10} | {s['Kept']:<6} | {s['Avg_Delta']:.4f}     | {s['Max_Delta']:.4f}")
            total_kept += s['Kept']
        print("-" * 65)
        print(f"Total Active Neurons: {total_kept}")
        print("=" * 65 + "\n")

    def get_stats_df(self) -> pd.DataFrame:
        """Returns training statistics as a pandas DataFrame."""
        return pd.DataFrame(self.training_stats)