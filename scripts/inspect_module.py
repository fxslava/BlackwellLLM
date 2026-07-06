import inspect
import os
import sys
from transformers import AutoModelForCausalLM


def main():
    print("==================================================")
    print(" Blackwell LLM: Module Source Code Inspector ")
    print("==================================================\n")

    # Корень чекпоинтов: BLACKWELL_MODELS_DIR (по умолчанию CWD — легаси ./llama3-8b-fp8)
    model_path = os.path.join(
        os.environ.get("BLACKWELL_MODELS_DIR", "."), "llama3-8b-fp8")
    print(f"[System] Быстрая загрузка структуры модели из '{model_path}'...\n")

    # Загружаем модель на CPU, нам нужна только структура питоновских классов
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cpu", low_cpu_mem_usage=True
    )

    # Добегаем до целевой проекции нулевого слоя
    q_proj = model.model.layers[0].self_attn.q_proj

    print(f"🎯 Целевой модуль: model.layers[0].self_attn.q_proj")
    print(f"📦 Имя класса:     {q_proj.__class__.__name__}")

    try:
        module_file = inspect.getfile(q_proj.__class__)
        print(f"📁 Файл библиотеки: {module_file}\n")
    except TypeError:
        print("📁 Файл библиотеки: [Встроенный или динамический класс]\n")

    print("--------------------------------------------------")
    print(" ИСХОДНЫЙ КОД МЕТОДА FORWARD:")
    print("--------------------------------------------------")

    try:
        # Вытаскиваем исходный код самого метода forward
        source_code = inspect.getsource(q_proj.forward)
        print(source_code)
    except Exception as e:
        print(f"[Ошибка] Не удалось извлечь код метода forward: {e}")
        print("\nПопытка извлечь исходный код всего класса...")
        try:
            class_source = inspect.getsource(q_proj.__class__)
            print(class_source)
        except Exception as ex:
            print(f"[Ошибка] Класс сгенерирован динамически: {ex}")


if __name__ == "__main__":
    main()