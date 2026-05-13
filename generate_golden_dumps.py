import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM

# =============================================================================
# 🕵️‍♂️ ШПИОНСКИЙ МОДУЛЬ: ПЕРЕХВАТ НА НИЗКОМ УРОВНЕ TORCH._SCALED_MM
# =============================================================================
orig_scaled_mm = torch._scaled_mm
current_target_name = None
captured_real_fp8_bytes = {}
captured_real_scales = {}


def spying_scaled_mm(
    input_a, input_b, scale_a, scale_b, out_dtype=None, **kwargs
):
    global current_target_name
    if current_target_name is not None:
        # 1. Захватываем истинные FP8 мантиссы активаций (массив по 1 байту)
        captured_real_fp8_bytes[f"{current_target_name}_input_fp8_bytes"] = (
            input_a.detach().clone().view(torch.uint8).cpu().numpy()
        )

        # 2. Захватываем истинный динамический скейл активаций, рассчитанный фреймворком
        captured_real_scales[f"{current_target_name}_input_scale"] = (
            scale_a.detach().clone().cpu().float().numpy()
        )

        print(
            f"  [Spy Intercept] Захвачены боевые FP8 байты и скейл для: '{current_target_name}'"
        )

    # Делегируем выполнение нативной функции для продолжения корректного прогона
    return orig_scaled_mm(
        input_a, input_b, scale_a, scale_b, out_dtype=out_dtype, **kwargs
    )


# Переопределяем указатель внутри PyTorch
torch._scaled_mm = spying_scaled_mm


# Фабрики контекстных переключателей для хуков
def enable_spy_pre_hook(target_name):
    def pre_hook(module, inputs):
        global current_target_name
        current_target_name = target_name

    return pre_hook


def disable_spy_post_hook():
    def post_hook(module, inputs, outputs):
        global current_target_name
        current_target_name = None

    return post_hook


# =============================================================================
# ГЛАВНЫЙ КОНВЕЙЕР ГЕНЕРАЦИИ ДАМПОВ
# =============================================================================
def main():
    print("==================================================")
    print(" Blackwell LLM: Automated Golden Dumps Generator ")
    print("==================================================\n")

    dumps_dir = "./dumps"
    os.makedirs(dumps_dir, exist_ok=True)
    print(f"[System] Cleared and prepared output directory: {dumps_dir}")

    model_path = "./llama3-8b-fp8"
    print(
        "[System] Loading FP8 model into VRAM (Bfloat16 precision baseline)..."
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Универсальная фабрика захвата стандартных вещественных тензоров
    def create_dump_hook(filename):
        def hook(module, input_tensor, output_tensor):
            tensor_data = (
                output_tensor[0]
                if isinstance(output_tensor, tuple)
                else output_tensor
            )
            filepath = os.path.join(dumps_dir, f"{filename}.bin")
            tensor_data.detach().cpu().float().numpy().tofile(filepath)

        return hook

    print(
        "[System] Registering forward hooks across all 32 transformer layers..."
    )

    # Дамп выхода таблицы эмбеддингов
    model.model.embed_tokens.register_forward_hook(create_dump_hook("embed_out"))

    # Послойная регистрация телеметрии
    for i, layer in enumerate(model.model.layers):
        # Нормализации
        layer.input_layernorm.register_forward_hook(
            create_dump_hook(f"layer_{i}_input_norm")
        )
        layer.post_attention_layernorm.register_forward_hook(
            create_dump_hook(f"layer_{i}_post_attn_norm")
        )

        # Выходы блоков
        layer.self_attn.register_forward_hook(create_dump_hook(f"layer_{i}_attn_out"))
        layer.mlp.register_forward_hook(create_dump_hook(f"layer_{i}_mlp_out"))
        layer.register_forward_hook(create_dump_hook(f"layer_{i}_accum_out"))

        # Выходы проекций
        layer.self_attn.q_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_q_proj")
        )
        layer.self_attn.k_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_k_proj")
        )
        layer.self_attn.v_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_v_proj")
        )
        layer.mlp.gate_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_gate_proj")
        )
        layer.mlp.up_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_up_proj")
        )

        # 🎯 ТОЧЕЧНЫЙ ПЕРЕХВАТ НА НУЛЕВОМ СЛОЕ: Включаем шпиона вокруг проекций
        if i == 0:
            # Проекции Attention (Q, K, V, O)
            for proj_name, target_module in [
                ("layer_0_q_proj", layer.self_attn.q_proj),
                ("layer_0_k_proj", layer.self_attn.k_proj),
                ("layer_0_v_proj", layer.self_attn.v_proj),
                ("layer_0_o_proj", layer.self_attn.o_proj),
                ("layer_0_gate_proj", layer.mlp.gate_proj),
                ("layer_0_up_proj", layer.mlp.up_proj),
                ("layer_0_down_proj", layer.mlp.down_proj),
            ]:
                target_module.register_forward_pre_hook(
                    enable_spy_pre_hook(proj_name)
                )
                target_module.register_forward_hook(disable_spy_post_hook())

        # Захват чистой математики внимания (ДО o_proj)
        def create_pre_dump_hook(filename):
            def pre_hook(module, input_tuple):
                tensor_data = input_tuple[0]
                filepath = os.path.join(dumps_dir, f"{filename}.bin")
                tensor_data.detach().cpu().float().numpy().tofile(filepath)

            return pre_hook

        layer.self_attn.o_proj.register_forward_pre_hook(
            create_pre_dump_hook(f"layer_{i}_attn_math")
        )

        if i == 31:
            layer.self_attn.o_proj.register_forward_hook(
                create_dump_hook("layer_31_post_attn")
            )

    # Финальные слои
    model.model.norm.register_forward_hook(create_dump_hook("final_norm_out"))
    model.lm_head.register_forward_hook(create_dump_hook("logits_out"))

    print("\n[System] Dumping Hugging Face decompressed baseline weights for Layer 0...")
    
    # Слой 0
    layer_0 = model.model.layers[0]
    
    # Словарь целевых весов для выгрузки
    target_weights = {
        "layer_0_q_proj_weight": layer_0.self_attn.q_proj.weight,
        "layer_0_k_proj_weight": layer_0.self_attn.k_proj.weight,
        "layer_0_v_proj_weight": layer_0.self_attn.v_proj.weight,
        "layer_0_gate_proj_weight": layer_0.mlp.gate_proj.weight,
        "layer_0_up_proj_weight": layer_0.mlp.up_proj.weight,
        "layer_0_down_proj_weight": layer_0.mlp.down_proj.weight,
    }

    # Сохраняем чистые вещественные матрицы на диск
    for name, weight_tensor in target_weights.items():
        filepath = os.path.join(dumps_dir, f"{name}.bin")
        # Приводим к float32 для безопасного чтения на стороне C++
        weight_tensor.detach().cpu().float().numpy().tofile(filepath)
        print(f"  [Weight Dump] Сохранена эталонная BF16 матрица: '{name}.bin' (shape: {list(weight_tensor.shape)})")

    # Запуск тестового прогона
    test_token_id = 128000
    print(f"\n[Engine] Executing Forward Pass for token_id: {test_token_id}...")

    dummy_input = torch.tensor([[test_token_id]], device="cuda:0")

    with torch.no_grad():
        output = model(dummy_input)

    next_token_id = output.logits[0, -1, :].argmax().item()

    # 💾 СБРОС ПЕРЕХВАЧЕННЫХ БОЕВЫХ БУФЕРОВ НА ДИСК
    print("\n[System] Flushing intercepted raw FP8 structures to disk...")
    for name, data_array in captured_real_fp8_bytes.items():
        filepath = os.path.join(dumps_dir, f"{name}.bin")
        data_array.tofile(filepath)

    for name, scale_array in captured_real_scales.items():
        filepath = os.path.join(dumps_dir, f"{name}.bin")
        scale_array.tofile(filepath)
        print(
            f"  [Reference Scale] {name}: {scale_array.flatten()[0]:.10f}"
        )

    print("--------------------------------------------------")
    print(f"[SUCCESS] All golden dumps successfully written to '{dumps_dir}'!")
    print(f"          Reference Input Token:  {test_token_id}")
    print(f"          Reference Output Token: {next_token_id}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()