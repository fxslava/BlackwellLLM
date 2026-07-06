import os
# 1. КРИТИЧЕСКИ ВАЖНО: Включаем динамические сегменты памяти ДО импорта torch.
# Это предотвращает фрагментацию VRAM и спасает от OOM в Windows.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import traceback
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

def main():
    print("==================================================")
    print(" Local Interactive FP8 Llama CLI Chat ")
    print("==================================================\n")

    if not torch.cuda.is_available():
        print("\n[CRITICAL ERROR]: PyTorch is not using CUDA!")
        return

    print(f"[Hardware] GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"[Hardware] VRAM used by OS: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB\n")

    # Checkpoint root: BLACKWELL_MODELS_DIR, defaulting to the CWD (the legacy
    # ./llama3-8b-fp8 layout).
    model_path = os.path.join(
        os.environ.get("BLACKWELL_MODELS_DIR", "."), "llama3-8b-fp8")

    print("[System] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("[System] Pushing FP8 weights directly to RTX 5070 VRAM...")
    
    # 2. ЖЕСТКАЯ ФИКСАЦИЯ ТИПА: требуем torch.bfloat16 вместо "auto".
    # Так как эмбеддинги лежат в BF16, это заставит PyTorch держать все 
    # промежуточные буферы в компактном 2-байтовом формате.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": "cuda:0"},
        torch_dtype=torch.bfloat16,  
        low_cpu_mem_usage=True
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    print("\n[System] Model loaded successfully! Active VRAM setup is ready.")
    print("[System] Type 'exit' or 'quit' to end the session.\n")
    print("--------------------------------------------------")

    messages = [
        {"role": "system", "content": "You are a highly intelligent, concise, and helpful AI assistant."}
    ]

    while True:
        try:
            user_input = input("\n[User]: ")
            
            if user_input.lower().strip() in ["exit", "quit"]:
                print("\n[System] Closing chat session. Goodbye!")
                break
            
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            # 1. Применяем шаблон (теперь inputs — это словарь BatchEncoding)
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            print("\n[Llama]: ", end="", flush=True)

            with torch.no_grad():
                # 2. КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: распаковываем ключи словаря через **inputs
                output_ids = model.generate(
                    **inputs,  
                    max_new_tokens=1024,
                    streamer=streamer,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            # 3. ВТОРОЕ КРИТИЧЕСКОЕ ИЗМЕНЕНИЕ: читаем .shape именно у тензора "input_ids"
            generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            messages.append({"role": "assistant", "content": response_text.strip()})

        except KeyboardInterrupt:
            print("\n\n[System] Current generation interrupted. Type 'exit' to quit.")
            continue
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print("\n\n[System Error]: VRAM limit exceeded. Context is too long.")
            continue
        except Exception as e:
            print(f"\n[Critical Error]: {repr(e)}")
            traceback.print_exc()
            break

if __name__ == "__main__":
    main()