import sys
from transformers import AutoTokenizer

def main():
    # Загружаем токенизатор из локальной папки твоей модели
    tokenizer = AutoTokenizer.from_pretrained("./llama3-8b-fp8")

    # Скрипт может принимать токены либо как аргумент консоли, либо просить ввести их
    if len(sys.argv) > 1:
        tokens_str = sys.argv[1]
    else:
        tokens_str = input("Вставь токены через пробел: ")

    if not tokens_str.strip():
        print("Пустой ввод. Выход.")
        return

    # 1. Разбиваем строку по пробелам и превращаем в массив целых чисел
    try:
        token_ids = [int(x) for x in tokens_str.strip().split()]
    except ValueError:
        print("Ошибка: строка должна содержать только числа, разделенные пробелами.")
        return

    print(f"\n[Decoder] Получен массив из {len(token_ids)} токенов.")

    # 2. Декодируем массив в текст
    # skip_special_tokens=False позволит увидеть служебные маркеры (например, <|begin_of_text|>)
    decoded_text = tokenizer.decode(token_ids, skip_special_tokens=False)

    print("-" * 50)
    print("РАСШИФРОВАННЫЙ ТЕКСТ:\n")
    print(decoded_text)
    print("\n" + "-" * 50)

if __name__ == "__main__":
    main()