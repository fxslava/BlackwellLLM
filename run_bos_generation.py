import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

def main():
    print("==================================================")
    print(" Blackwell LLM: BOS Token Generation (ID 128000) ")
    print("==================================================\n")

    model_path = "./llama3-8b-fp8"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("[System] Loading FP8 model into VRAM...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Включаем стример, чтобы видеть появление каждого токена в реальном времени
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

    # Официальный стартовый якорь начала текста
    bos_token_id = 128000
    inputs = torch.tensor([[bos_token_id]], device="cuda:0")

    print(f"\n[Engine] Streaming response starting from BOS token ({bos_token_id})...\n")
    print("--------------------------------------------------")

    with torch.no_grad():
        model.generate(
            inputs,
            max_new_tokens=150,  # Генерируем 150 токенов для оценки семантики
            streamer=streamer,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    print("\n--------------------------------------------------\n")

if __name__ == "__main__":
    main()