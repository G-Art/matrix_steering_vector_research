import torch
import torch.nn.functional as F
import contextlib
from typing import Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from dsv.steereng_vector_v3 import MatrixSteeringVector

# ==========================================
# 1. UTILS & EXTRACTION HOOKS / УТИЛІТИ ТА ХУКИ ВИДОБУТКУ
# ==========================================

@contextlib.contextmanager
def register_hooks(hooks_list):
    """
    Context manager to safely register and remove hooks.

    Менеджер контексту для безпечної реєстрації та видалення хуків.
    """
    try:
        yield
    finally:
        for handle in hooks_list:
            handle.remove()

# ==========================================
# 2. FULL-CONTEXT MATRIX HOOK
# ==========================================
class MatrixSteeringHook:
    """
    Hook that applies affine transformation to the hidden states: Output = Input + M * (Target - Input).
    Target is calculated as: Input @ W.T + B.

    Хук, що застосовує афінне перетворення до прихованих станів.
    Цільове значення розраховується як: Вхід @ W.T + B.
    """

    def __init__(self, vector_obj: MatrixSteeringVector, m: float, device: torch.device):
        self.device = device
        self.M = m
        self.layer_idx = vector_obj.layer_index
        self.indices = torch.tensor(vector_obj.active_indices, device=device, dtype=torch.long)

        # W shape: [N_active, Hidden_Size], B shape: [N_active]
        self.weight = torch.tensor(vector_obj.weight_matrix, device=device, dtype=torch.float)
        self.bias = torch.tensor(vector_obj.bias_vector, device=device, dtype=torch.float)

    def __call__(self, module, args):
        x = args[0]  # [Batch, Seq, Hidden]

        # Calculate target for active neurons based on full context
        # Розрахунок цільового значення для активних нейронів на основі повного контексту
        target_active = F.linear(x, self.weight, self.bias)

        current_active = x[..., self.indices]
        delta = target_active - current_active

        # Create sparse update tensor
        # Створення розрідженого тензора оновлення
        full_delta = torch.zeros_like(x)
        full_delta[..., self.indices] = delta

        return (x + (full_delta * self.M),)


def generate_with_advanced_steering(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        text: str,
        trained_vectors: Dict[int, MatrixSteeringVector],
        mix_strength_M: float,
        generation_config: Dict[str, Any]
) -> str:

    model.eval()
    device = next(model.parameters()).device
    hook_handles = []

    try:
        for layer_idx, vector_obj in trained_vectors.items():

            hook = MatrixSteeringHook(
                vector_obj=vector_obj,
                m=mix_strength_M,
                device=device,
                # limit_value=limit_value,
                # start_boost=start_boost
            )
            handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(hook)
            hook_handles.append(handle)

        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad(), register_hooks(hook_handles):
            outputs = model.generate(**inputs, **generation_config)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"
    finally:
        for handle in hook_handles: handle.remove()