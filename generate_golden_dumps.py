import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def main():
    print("==================================================")
    print(" Blackwell LLM: Automated Golden Dumps Generator ")
    print("==================================================\n")

    dumps_dir = "./dumps"
    os.makedirs(dumps_dir, exist_ok=True)
    print(f"[System] Cleared and prepared output directory: {dumps_dir}")

    model_path = "./llama3-8b-fp8"
    print("[System] Loading FP8 model into VRAM (Bfloat16 precision baseline)...")

    # Загружаем модель строго в BF16, чтобы данные активаций совпадали с нашим конвейером
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # 1. Универсальная фабрика перехватчиков выходов (хуков)
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

    # 🎯 2. НОВОЕ: Фабрика Pre-хуков для захвата и расчета эталонных динамических скейлов
    def create_dynamic_scale_pre_hook(filename):
        def pre_hook(module, input_tuple):
            x = input_tuple[0]  # Входящие активации: [batch, seq_len, dim]

            # 🎯 Кастуем amax в float32 ДО деления, чтобы отключить авто-каст PyTorch в BF16
            amax = x.abs().amax(dim=-1, keepdim=True).float()

            # Теперь scale рассчитывается и сохраняется в кристально чистом Float32
            scale = amax / 448.0
            scale = torch.clamp(scale, min=1e-12)

            filepath = os.path.join(dumps_dir, f"{filename}.bin")
            scale_val = scale.detach().cpu().float().numpy()
            scale_val.tofile(filepath)

            print(f"  [Scale Dump] Захвачен истинный Float32 скейл '{filename}': {scale_val.flatten()[0]:.10f}")

        return pre_hook

    print("[System] Registering forward hooks across all 32 transformer layers...")

    # Дамп выхода таблицы эмбеддингов
    model.model.embed_tokens.register_forward_hook(create_dump_hook("embed_out"))

    # Послойные дампы (Внимание, MLP и полный остаточный поток)
    for i, layer in enumerate(model.model.layers):
        # Нормализация
        layer.input_layernorm.register_forward_hook(
            create_dump_hook(f"layer_{i}_input_norm")
        )
        layer.post_attention_layernorm.register_forward_hook(
            create_dump_hook(f"layer_{i}_post_attn_norm")
        )

        # Стандартные выходы блоков
        layer.self_attn.register_forward_hook(create_dump_hook(f"layer_{i}_attn_out"))
        layer.mlp.register_forward_hook(create_dump_hook(f"layer_{i}_mlp_out"))
        layer.register_forward_hook(create_dump_hook(f"layer_{i}_accum_out"))

        # Гранулярные проекции Attention
        layer.self_attn.q_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_q_proj")
        )
        layer.self_attn.k_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_k_proj")
        )
        layer.self_attn.v_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_v_proj")
        )

        # 🎯 Захватываем динамические скейлы активаций для Layer 0
        if i == 0:
            # Скейл для входа в Q, K, V (рассчитывается по d_X_norm)
            layer.self_attn.q_proj.register_forward_pre_hook(
                create_dynamic_scale_pre_hook("layer_0_qkv_input_scale")
            )
            # Скейл для входа в выходную проекцию внимания (рассчитывается по d_Attn_out)
            layer.self_attn.o_proj.register_forward_pre_hook(
                create_dynamic_scale_pre_hook("layer_0_o_proj_input_scale")
            )
            # Скейл для входа в гейты MLP
            layer.mlp.gate_proj.register_forward_pre_hook(
                create_dynamic_scale_pre_hook("layer_0_mlp_input_scale")
            )
            # Скейл для входа в down_proj (рассчитывается по d_Swiglu_out)
            layer.mlp.down_proj.register_forward_pre_hook(
                create_dynamic_scale_pre_hook("layer_0_down_proj_input_scale")
            )

        # Pre-hook: захват чистой математики внимания (ДО o_proj)
        def create_pre_dump_hook(filename):
            def pre_hook(module, input_tuple):
                tensor_data = input_tuple[0]
                filepath = os.path.join(dumps_dir, f"{filename}.bin")
                tensor_data.detach().cpu().float().numpy().tofile(filepath)

            return pre_hook

        layer.self_attn.o_proj.register_forward_pre_hook(
            create_pre_dump_hook(f"layer_{i}_attn_math")
        )

        # Точечный дамп остаточного потока сразу после Attention для Layer 31
        if i == 31:
            layer.self_attn.o_proj.register_forward_hook(
                create_dump_hook("layer_31_post_attn")
            )

        # Гранулярные проекции MLP
        layer.mlp.gate_proj.register_forward_hook(
            create_dump_hook(f"layer_{i}_gate_proj")
        )
        layer.mlp.up_proj.register_forward_hook(create_dump_hook(f"layer_{i}_up_proj"))

    # Финальная нормализация
    model.model.norm.register_forward_hook(create_dump_hook("final_norm_out"))

    # Финальные логиты словаря (lm_head)
    model.lm_head.register_forward_hook(create_dump_hook("logits_out"))

    # Запуск прогона
    test_token_id = 128000
    print(f"\n[Engine] Executing Forward Pass for token_id: {test_token_id}...")

    dummy_input = torch.tensor([[test_token_id]], device="cuda:0")

    with torch.no_grad():
        output = model(dummy_input)

    next_token_id = output.logits[0, -1, :].argmax().item()

    print("--------------------------------------------------")
    print(f"[SUCCESS] All golden dumps successfully written to '{dumps_dir}'!")
    print(f"          Reference Input Token:  {test_token_id}")
    print(f"          Reference Output Token: {next_token_id}")
    print("--------------------------------------------------\n")


if __name__ == "__main__":
    main()