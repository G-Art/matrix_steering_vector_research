from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from dsv.steering_hook_v3 import MatrixSteeringHook


# ==========================================
# 1. PASSIVE HOOK (No changes)
# ==========================================
class PassiveHook:
    def __init__(self, layer_idx, storage, neurons_of_interest, run_name="Ref"):
        self.layer_idx = layer_idx
        self.storage = storage
        self.neurons = neurons_of_interest
        self.run_name = run_name
        self.step_counter = 0

    def __call__(self, module, args):
        x = args[0]
        is_prompt = x.shape[1] > 1
        current_step = "Prompt" if is_prompt else self.step_counter

        if x.dim() == 3:
            last_token_vec = x[0, -1, :].detach().cpu()
        else:
            last_token_vec = x[-1, :].detach().cpu()

        for idx in self.neurons:
            self.storage[(self.run_name, self.layer_idx, idx, current_step)] = last_token_vec[idx].item()

        if not is_prompt:
            self.step_counter += 1
        return None


# ==========================================
# 2. MATRIX DEBUG HOOK (Refined)
# ==========================================
class MatrixDebugHookWrapper:
    def __init__(self, layer_idx: int, real_hook, log_storage: List[Dict], ref_db: Dict):
        self.layer_idx = layer_idx
        self.hook = real_hook
        self.storage = log_storage
        self.ref_db = ref_db
        self.step_counter = 0
        self.active_indices = self.hook.indices.cpu().numpy()

    def __call__(self, module, args):
        x_in = args[0]
        is_prompt = x_in.shape[1] > 1
        current_step_label = "Prompt" if is_prompt else self.step_counter

        # 1. Capture Input (This is the Distorted Input in the Steered Run)
        if x_in.dim() == 3:
            last_token_in = x_in[0, -1, :].detach().cpu()
        else:
            last_token_in = x_in[-1, :].detach().cpu()

        # 2. Execute Real Matrix Hook
        x_out_tuple = self.hook(module, args)
        x_out = x_out_tuple[0]

        # 3. Capture Output (Steered)
        if x_out.dim() == 3:
            last_token_out = x_out[0, -1, :].detach().cpu()
        else:
            last_token_out = x_out[-1, :].detach().cpu()

        # 4. Log Data
        for idx in self.active_indices:
            idx = int(idx)
            val_in = last_token_in[idx].item()
            val_out = last_token_out[idx].item()

            ref_target = self.ref_db.get(("TARGET", self.layer_idx, idx, current_step_label), 0.0)
            ref_base = self.ref_db.get(("BASE", self.layer_idx, idx, current_step_label), 0.0)

            record = {
                "Step": current_step_label,
                "Layer": self.layer_idx,
                "NeuronID": idx,
                "Input_Distorted": val_in,  # Where we started this layer
                "Input_Pure_BASE": ref_base,  # Where we would be without any steering
                "Reference_Pure_TARGET": ref_target,  # Where we want to be (Target)
                "Output_Steered": val_out,  # Where we ended up
            }
            self.storage.append(record)

        if not is_prompt:
            self.step_counter += 1
        return x_out_tuple


# ==========================================
# 3. COMPARATIVE DEBUGGER (Same logic)
# ==========================================
class ComparativeMatrixDebugger:
    def __init__(self, model, tokenizer, vectors):
        self.model = model
        self.tokenizer = tokenizer
        self.vectors = vectors
        self.logs = []
        self.ref_db = {}

    def run_trace(self, prompt_base, prompt_target, mix_strength=1.0, max_tokens=10):
        self.logs = []
        self.ref_db = {}
        neurons_of_interest = {l: vec.active_indices for l, vec in self.vectors.items()}

        # PASS 1: Pure Target
        print(f"🔬 Pass 1: Pure Target (Target Collection)")
        hooks = [
            self.model.model.layers[l].mlp.down_proj.register_forward_pre_hook(PassiveHook(l, self.ref_db, ids, "TARGET"))
            for l, ids in neurons_of_interest.items()]
        inputs = self.tokenizer(prompt_target, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        for h in hooks: h.remove()

        # PASS 2: Pure Base (Baseline)
        print(f"🔬 Pass 2: Pure Base (Baseline Collection)")
        hooks = [
            self.model.model.layers[l].mlp.down_proj.register_forward_pre_hook(PassiveHook(l, self.ref_db, ids, "BASE"))
            for l, ids in neurons_of_interest.items()]
        inputs = self.tokenizer(prompt_base, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        for h in hooks: h.remove()

        # PASS 3: Steered
        print(f"🔬 Pass 3: Matrix Steered Generation")
        hooks = []
        for layer_idx, vector_obj in self.vectors.items():
            real_hook = MatrixSteeringHook(vector_obj, mix_strength, self.model.device)
            debug_hook = MatrixDebugHookWrapper(layer_idx, real_hook, self.logs, self.ref_db)
            hooks.append(self.model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(debug_hook))

        inputs = self.tokenizer(prompt_base, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)

        gen_text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"🤖 Result: {gen_text}")
        for h in hooks: h.remove()

        return pd.DataFrame(self.logs)


# ==========================================
# 4. NEW VISUALIZATION SUITE
# ==========================================
class Visualizer:
    @staticmethod
    def _prepare_vectors(df, step):
        """Helper to pivot data into vectors for math operations."""
        # Filter by step
        step_data = df[df["Step"] == step].copy()

        # Group by layer
        layer_groups = step_data.groupby("Layer")
        results = {}

        for layer, group in layer_groups:
            # Sort by NeuronID to ensure vector alignment
            group = group.sort_values("NeuronID")

            # Vectors (numpy arrays)
            v_in_distorted = group["Input_Distorted"].values
            v_target_fr = group["Reference_Pure_TARGET"].values
            v_out_steered = group["Output_Steered"].values

            results[layer] = {
                "distorted": v_in_distorted,
                "target": v_target_fr,
                "steered": v_out_steered,
                "neurons": group["NeuronID"].values
            }
        return results

    @staticmethod
    def plot_layer_fidelity(df, step="Prompt"):
        """

        Plots how well the steering aligns with the ideal direction per layer.
        """
        data = Visualizer._prepare_vectors(df, step)
        layers = sorted(data.keys())
        similarities = []
        magnitudes_ideal = []
        magnitudes_actual = []

        for l in layers:
            d = data[l]
            # Ideal Vector: From current input to pure French
            vec_ideal = d["target"] - d["distorted"]
            # Actual Vector: From current input to steered output
            vec_actual = d["steered"] - d["distorted"]

            # Cosine Sim
            norm_ideal = np.linalg.norm(vec_ideal) + 1e-9
            norm_actual = np.linalg.norm(vec_actual) + 1e-9
            cos_sim = np.dot(vec_ideal, vec_actual) / (norm_ideal * norm_actual)

            similarities.append(cos_sim)
            magnitudes_ideal.append(norm_ideal)
            magnitudes_actual.append(norm_actual)

        # Plotting
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar plot for Similarity
        colors = ['#2ca02c' if s > 0.8 else '#d62728' for s in similarities]
        bars = ax1.bar(layers, similarities, color=colors, alpha=0.7, label='Cosine Similarity')
        ax1.set_xlabel('Layer Depth')
        ax1.set_ylabel('Directional Fidelity (Cosine Sim)', color='#2ca02c')
        ax1.set_ylim(-0.1, 1.1)
        ax1.axhline(1.0, color='grey', linestyle='--', alpha=0.5)
        ax1.axhline(0.0, color='grey', linewidth=0.5)

        # Line plot for Magnitude Ratio on secondary axis
        ax2 = ax1.twinx()
        ratios = np.array(magnitudes_actual) / np.array(magnitudes_ideal)
        ax2.plot(layers, ratios, color='blue', marker='o', linestyle='-', linewidth=2,
                 label='Force Ratio (Actual/Ideal)')
        ax2.set_ylabel('Force Magnitude Ratio', color='blue')
        ax2.set_ylim(0, max(ratios) * 1.2)

        plt.title(f"Steering Fidelity Analysis (Step: {step})")

        # Legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_vector_shift_2d(df, layer, step="Prompt"):
        """


[Image of Vector Projection]

        Projects the high-dimensional shift onto a 2D plane to visualize the trajectory.
        """
        data = Visualizer._prepare_vectors(df, step)
        if layer not in data:
            print(f"Layer {layer} not found in logs.")
            return

        d = data[layer]

        subset = df[(df["Layer"] == layer) & (df["Step"] == step)].sort_values("NeuronID")
        v_pure_en = subset["Input_Pure_BASE"].values

        v_start = d["distorted"]
        v_target = d["target"]
        v_steered = d["steered"]

        # Stack vectors for PCA
        # We want to find the best 2D plane that explains the variance between these 4 points
        matrix = np.vstack([v_start, v_target, v_steered, v_pure_en])

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(matrix)

        p_start = pca_result[0]
        p_target = pca_result[1]
        p_steered = pca_result[2]
        p_base = pca_result[3]

        plt.figure(figsize=(10, 8))

        # Plot Points
        plt.scatter(p_start[0], p_start[1], color='orange', s=50, label='Input (Distorted)', zorder=5)
        plt.scatter(p_target[0], p_target[1], color='green', marker='*', s=100, label='Ideal Target', zorder=5)
        plt.scatter(p_steered[0], p_steered[1], color='blue', s=50, label='Steered Output', zorder=5)
        plt.scatter(p_base[0], p_base[1], color='red', marker='x', s=50, label='Base', zorder=5)

        # Plot Arrows (Trajectories)
        # 1. Ideal Path: Start -> Target
        plt.arrow(p_start[0], p_start[1], p_target[0] - p_start[0], p_target[1] - p_start[1],
                  fc='green', ec='green', alpha=0.2, width=0.03, linestyle='--', length_includes_head=True)

        # 2. Actual Steering: Start -> Steered
        plt.arrow(p_start[0], p_start[1], p_steered[0] - p_start[0], p_steered[1] - p_start[1],
                  fc='blue', ec='blue', alpha=0.8, width=0.05, length_includes_head=True, label='Applied Vector')

        # 3. Drift (The error): Steered -> Target
        plt.plot([p_steered[0], p_target[0]], [p_steered[1], p_target[1]], 'r:', label='Residual Error')

        plt.title(f"2D Vector Projection (Layer {layer})")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} var)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} var)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

# ==========================================
# EXAMPLE USAGE
# ==========================================
# debugger = ComparativeMatrixDebugger(model, tokenizer, trained_vectors)
# df_logs = debugger.run_trace("Hello world", "Bonjour le monde", mix_strength=1.0)
#
# # 1. Show global convergence
# Visualizer.plot_layer_fidelity(df_logs, step="Prompt")
#
# # 2. Drill down into specific layer geometry
# Visualizer.plot_vector_shift_2d(df_logs, layer=15, step="Prompt")