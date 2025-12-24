import os
import json
import base64
import random
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


STANDARD_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
DEFAULT_PAD = "="
PHRASE_SEPARATOR = " "

DEFAULT_ALPHABET_FILE = Path(__file__).with_name("alphabet_example.json")
DEFAULT_PHRASE_FILE = Path(__file__).with_name("phrase_profiles_example.json")


def _extract_alphabet_profiles(data: object) -> dict[str, tuple[str, str]]:
    if isinstance(data, dict) and "profiles" in data:
        profiles = data.get("profiles", [])
    elif isinstance(data, list):
        profiles = data
    else:
        raise ValueError("JSON 顶层需为数组，或包含 profiles 数组")

    result: dict[str, tuple[str, str]] = {}
    for item in profiles:
        name = item.get("name") if isinstance(item, dict) else None
        alphabet = item.get("alphabet") if isinstance(item, dict) else None
        pad = item.get("pad", DEFAULT_PAD) if isinstance(item, dict) else DEFAULT_PAD
        if not name or not alphabet:
            continue
        if len(alphabet) != 64 or len(pad) != 1:
            continue
        result[name] = (alphabet, pad)

    if not result:
        raise ValueError("未找到有效的字符表条目 (需含 name 与 64 长度 alphabet)")
    return result


def _extract_phrase_profiles(
    data: object,
) -> dict[str, tuple[dict[int, tuple[list[str], list[int]]], dict[str, int], set[str]]]:
    if isinstance(data, dict) and "phrase_profiles" in data:
        profiles = data.get("phrase_profiles", [])
    elif isinstance(data, list):
        profiles = data
    else:
        raise ValueError("JSON 顶层需为数组，或包含 phrase_profiles 数组")

    result: dict[
        str, tuple[dict[int, tuple[list[str], list[int]]], dict[str, int], set[str]]
    ] = {}
    for item in profiles:
        name = item.get("name") if isinstance(item, dict) else None
        if not name:
            continue
        vt, pt, pf = _build_phrase_tables(item)
        result[name] = (vt, pt, pf)

    if not result:
        raise ValueError("未找到有效的短语表条目 (需 name 和 0-63 的 phrases)")
    return result


def _auto_load_alphabets() -> dict[str, tuple[str, str]]:
    if not DEFAULT_ALPHABET_FILE.exists():
        return {}
    try:
        with open(DEFAULT_ALPHABET_FILE, "r", encoding="utf-8") as f:
            return _extract_alphabet_profiles(json.load(f))
    except Exception:
        return {}


def _auto_load_phrase_profiles() -> dict[
    str, tuple[dict[int, tuple[list[str], list[int]]], dict[str, int], set[str]]
]:
    if not DEFAULT_PHRASE_FILE.exists():
        return {}
    try:
        with open(DEFAULT_PHRASE_FILE, "r", encoding="utf-8") as f:
            return _extract_phrase_profiles(json.load(f))
    except Exception:
        return {}


def _derive_key(key_hex: str) -> bytes:
    """解析十六进制密钥并校验长度（16/24/32 字节）。"""
    key = bytes.fromhex(key_hex.strip())
    if len(key) not in (16, 24, 32):
        raise ValueError("密钥需为 128/192/256 位（32/48/64 个十六进制字符）")
    return key


def _pkcs7_pad(data: bytes) -> bytes:
    padder = padding.PKCS7(128).padder()
    return padder.update(data) + padder.finalize()


def _pkcs7_unpad(data: bytes) -> bytes:
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(data) + unpadder.finalize()


def _encode_custom(blob: bytes, alphabet: str, pad: str = DEFAULT_PAD) -> str:
    if len(alphabet) != 64:
        raise ValueError("自定义字符表需包含 64 个字符")
    if len(pad) != 1:
        raise ValueError("填充符需为单个字符")
    encoded = base64.b64encode(blob).decode("ascii")
    table = str.maketrans(STANDARD_ALPHABET + DEFAULT_PAD, alphabet + pad)
    return encoded.translate(table)


def _decode_custom(text: str, alphabet: str, pad: str = DEFAULT_PAD) -> bytes:
    if len(alphabet) != 64:
        raise ValueError("自定义字符表需包含 64 个字符")
    if len(pad) != 1:
        raise ValueError("填充符需为单个字符")
    table = str.maketrans(alphabet + pad, STANDARD_ALPHABET + DEFAULT_PAD)
    standard = text.translate(table)
    return base64.b64decode(standard)


def _build_phrase_tables(
    profile: dict,
) -> tuple[dict[int, tuple[list[str], list[int]]], dict[str, int], set[str]]:
    entries = profile.get("entries", [])
    value_to_choices: dict[int, tuple[list[str], list[int]]] = {}
    phrase_to_value: dict[str, int] = {}
    prefix_set: set[str] = set()

    for entry in entries:
        value = entry.get("value")
        phrases = entry.get("phrases", [])
        if not isinstance(value, int) or not (0 <= value <= 63):
            continue
        texts: list[str] = []
        weights: list[int] = []
        for item in phrases:
            if not isinstance(item, list) or len(item) != 2:
                continue
            phrase, weight = item
            if not phrase or phrase in phrase_to_value:
                continue
            w = int(weight) if isinstance(weight, (int, float)) else 1
            if w <= 0:
                w = 1
            phrase_to_value[phrase] = value
            texts.append(str(phrase))
            weights.append(w)
            # 前缀集合（不含空串）
            for i in range(1, len(phrase)):
                prefix_set.add(phrase[:i])
        if texts:
            value_to_choices[value] = (texts, weights)

    # 确保 0~63 都有定义
    missing = [v for v in range(64) if v not in value_to_choices]
    if missing:
        raise ValueError(f"短语表缺少值: {missing}")
    return value_to_choices, phrase_to_value, prefix_set


def _phrase_encode_from_base64(
    b64_text: str, value_to_choices: dict[int, tuple[list[str], list[int]]]
) -> str:
    tokens: list[str] = []
    for ch in b64_text:
        if ch == DEFAULT_PAD:
            tokens.append(DEFAULT_PAD)
            continue
        idx = STANDARD_ALPHABET.index(ch)
        phrases, weights = value_to_choices[idx]
        tokens.append(random.choices(phrases, weights=weights, k=1)[0])
    return "".join(tokens)


def _phrase_decode_to_base64(
    text: str, phrase_to_value: dict[str, int], prefix_set: set[str]
) -> str:
    chars: list[str] = []
    buffer = ""
    best_match: str | None = None

    def flush_best() -> None:
        nonlocal buffer, best_match
        if best_match is None:
            raise ValueError(f"未识别短语: {buffer}")
        val = phrase_to_value[best_match]
        chars.append(STANDARD_ALPHABET[val])
        # 将溢出的部分作为新起点
        buffer = ""
        best_match = None

    for ch in text:
        if ch == DEFAULT_PAD:
            # 遇到填充符，先结束前面的短语
            if buffer:
                if buffer in phrase_to_value:
                    val = phrase_to_value[buffer]
                    chars.append(STANDARD_ALPHABET[val])
                    buffer = ""
                    best_match = None
                else:
                    flush_best()
            chars.append(DEFAULT_PAD)
            continue

        buffer += ch
        if buffer in phrase_to_value:
            best_match = buffer
        if (buffer not in prefix_set) and (buffer not in phrase_to_value):
            # 当前 buffer 不是任何前缀/短语，回退到上一个最佳匹配
            if best_match is None:
                raise ValueError(f"未识别短语: {buffer}")
            val = phrase_to_value[best_match]
            chars.append(STANDARD_ALPHABET[val])
            # 重新开始，从当前字符作为新起点
            buffer = ch
            best_match = buffer if buffer in phrase_to_value else None
            if (buffer not in prefix_set) and (buffer not in phrase_to_value):
                raise ValueError(f"未识别短语: {buffer}")

    # 结束时处理残余
    if buffer:
        if buffer in phrase_to_value:
            val = phrase_to_value[buffer]
            chars.append(STANDARD_ALPHABET[val])
        elif best_match:
            val = phrase_to_value[best_match]
            chars.append(STANDARD_ALPHABET[val])
        else:
            raise ValueError(f"未识别短语: {buffer}")

    return "".join(chars)


def encrypt_bytes(key_hex: str, data: bytes) -> bytes:
    key = _derive_key(key_hex)
    iv = os.urandom(16)
    padded = _pkcs7_pad(data)

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    return iv + ciphertext


def decrypt_bytes(key_hex: str, blob: bytes) -> bytes:
    key = _derive_key(key_hex)
    if len(blob) < 16:
        raise ValueError("密文长度不足（缺少 IV）")
    iv, ciphertext = blob[:16], blob[16:]

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    return _pkcs7_unpad(padded)


def encrypt_file(key_hex: str, plaintext_path: Path) -> Path:
    plaintext = plaintext_path.read_bytes()
    blob = encrypt_bytes(key_hex, plaintext)

    out_path = plaintext_path.with_suffix(plaintext_path.suffix + ".enc")
    out_path.write_bytes(blob)
    return out_path


def decrypt_file(key_hex: str, ciphertext_path: Path) -> Path:
    blob = ciphertext_path.read_bytes()
    plaintext = decrypt_bytes(key_hex, blob)

    if ciphertext_path.suffix == ".enc":
        out_path = ciphertext_path.with_suffix("")
    else:
        out_path = ciphertext_path.with_suffix(ciphertext_path.suffix + ".dec")
    out_path.write_bytes(plaintext)
    return out_path


class AESApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("AES 文件加解密工具")
        root.resizable(False, False)

        self.key_var = tk.StringVar()
        self.plain_path_var = tk.StringVar()
        self.cipher_path_var = tk.StringVar()
        self.alphabets: dict[str, tuple[str, str]] = {
            "Base64 (标准)": (STANDARD_ALPHABET, DEFAULT_PAD)
        }
        self.alphabets.update(_auto_load_alphabets())
        self.alphabet_choice = tk.StringVar(value=list(self.alphabets.keys())[0])

        # 短语模式
        self.mode_var = tk.StringVar(value="alphabet")  # alphabet | phrase
        self.phrase_profiles: dict[
            str, tuple[dict[int, tuple[list[str], list[int]]], dict[str, int], set[str]]
        ] = {}
        self.phrase_profiles.update(_auto_load_phrase_profiles())
        self.phrase_choice = tk.StringVar(value=next(iter(self.phrase_profiles), ""))

        self._build_key_frame()
        self._build_encrypt_frame()
        self._build_decrypt_frame()
        self._build_text_frame()

    def _build_key_frame(self) -> None:
        frame = tk.LabelFrame(self.root, text="密钥（十六进制）", padx=10, pady=10)
        frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        entry = tk.Entry(frame, textvariable=self.key_var, width=64)
        entry.grid(row=0, column=0, padx=(0, 8))

        tk.Button(frame, text="生成 256 位密钥", command=self._generate_key).grid(
            row=0, column=1, padx=(0, 6)
        )
        tk.Button(frame, text="复制密钥", command=self._copy_key).grid(row=0, column=2)

        tk.Label(
            frame,
            text="支持 32/48/64 个十六进制字符（AES-128/192/256）",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_encrypt_frame(self) -> None:
        frame = tk.LabelFrame(self.root, text="加密", padx=10, pady=10)
        frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        tk.Button(frame, text="选择明文文件", command=self._select_plain).grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(frame, textvariable=self.plain_path_var, width=60, anchor="w").grid(
            row=0, column=1, padx=(8, 0)
        )

        tk.Button(frame, text="开始加密", command=self._encrypt_handler).grid(
            row=1, column=0, pady=(8, 0)
        )

    def _build_decrypt_frame(self) -> None:
        frame = tk.LabelFrame(self.root, text="解密", padx=10, pady=10)
        frame.grid(row=2, column=0, padx=10, pady=(5, 10), sticky="ew")

        tk.Button(frame, text="选择密文文件", command=self._select_cipher).grid(
            row=0, column=0, sticky="w"
        )
        tk.Label(frame, textvariable=self.cipher_path_var, width=60, anchor="w").grid(
            row=0, column=1, padx=(8, 0)
        )

        tk.Button(frame, text="开始解密", command=self._decrypt_handler).grid(
            row=1, column=0, pady=(8, 0)
        )

    def _build_text_frame(self) -> None:
        frame = tk.LabelFrame(self.root, text="文本加解密 (Base64 输出)", padx=10, pady=10)
        frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")

        tk.Label(frame, text="明文：").grid(row=0, column=0, sticky="nw")
        self.plain_text = tk.Text(frame, width=60, height=6)
        self.plain_text.grid(row=0, column=1, padx=(6, 0))

        tk.Label(frame, text="密文 (Base64)：").grid(row=1, column=0, sticky="nw", pady=(6, 0))
        self.cipher_text = tk.Text(frame, width=60, height=10)
        self.cipher_text.grid(row=1, column=1, padx=(6, 0), pady=(6, 0))

        alpha_frame = tk.Frame(frame)
        alpha_frame.grid(row=2, column=1, sticky="w", pady=(8, 0))
        tk.Label(alpha_frame, text="模式：").grid(row=0, column=0, sticky="w")
        tk.Radiobutton(
            alpha_frame,
            text="字符表",
            variable=self.mode_var,
            value="alphabet",
        ).grid(row=0, column=1, padx=(0, 6))
        tk.Radiobutton(
            alpha_frame,
            text="短语表",
            variable=self.mode_var,
            value="phrase",
        ).grid(row=0, column=2, padx=(0, 6))

        tk.Label(alpha_frame, text="字符表：").grid(row=1, column=0, sticky="w")
        self.alpha_menu = tk.OptionMenu(alpha_frame, self.alphabet_choice, *self.alphabets.keys())
        self.alpha_menu.grid(row=1, column=1, padx=(4, 6))
        tk.Button(alpha_frame, text="加载字符表 JSON", command=self._load_alphabet_json).grid(
            row=1, column=2
        )

        tk.Label(alpha_frame, text="短语表：").grid(row=2, column=0, sticky="w", pady=(4, 0))
        self.phrase_menu = tk.OptionMenu(
            alpha_frame,
            self.phrase_choice,
            *(self.phrase_profiles.keys() if self.phrase_profiles else ("",)),
        )
        self.phrase_menu.grid(row=2, column=1, padx=(4, 6), pady=(4, 0))
        tk.Button(alpha_frame, text="加载短语表 JSON", command=self._load_phrase_json).grid(
            row=2, column=2, pady=(4, 0)
        )

        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=4, column=1, sticky="w", pady=(8, 0))
        tk.Button(btn_frame, text="明文加密 →", command=self._encrypt_text_handler).grid(
            row=0, column=0, padx=(0, 6)
        )
        tk.Button(btn_frame, text="← 密文解密", command=self._decrypt_text_handler).grid(
            row=0, column=1
        )

    def _generate_key(self) -> None:
        self.key_var.set(os.urandom(32).hex())

    def _copy_key(self) -> None:
        key = self.key_var.get().strip()
        if not key:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(key)

    def _load_alphabet_json(self) -> None:
        path = filedialog.askopenfilename(
            title="选择包含字符表的 JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = _extract_alphabet_profiles(json.load(f))
            self.alphabets.update(loaded)
            added = len(loaded)

            menu = self.alpha_menu["menu"]
            menu.delete(0, "end")
            for key in self.alphabets.keys():
                menu.add_command(label=key, command=lambda v=key: self.alphabet_choice.set(v))
            # 自动选择最新添加的一个
            self.alphabet_choice.set(list(self.alphabets.keys())[-1])
            messagebox.showinfo("成功", f"已加载 {added} 个字符表")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", str(exc))

    def _load_phrase_json(self) -> None:
        path = filedialog.askopenfilename(
            title="选择包含短语表的 JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                loaded = _extract_phrase_profiles(json.load(f))
            self.phrase_profiles.update(loaded)
            added = len(loaded)

            menu = self.phrase_menu["menu"]
            menu.delete(0, "end")
            for key in self.phrase_profiles.keys():
                menu.add_command(label=key, command=lambda v=key: self.phrase_choice.set(v))
            self.phrase_choice.set(list(self.phrase_profiles.keys())[-1])
            messagebox.showinfo("成功", f"已加载 {added} 个短语表")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", str(exc))

    def _select_plain(self) -> None:
        path = filedialog.askopenfilename(title="Choose plaintext file")
        if path:
            self.plain_path_var.set(path)

    def _select_cipher(self) -> None:
        path = filedialog.askopenfilename(title="Choose ciphertext file")
        if path:
            self.cipher_path_var.set(path)

    def _encrypt_handler(self) -> None:
        try:
            if not self.key_var.get().strip():
                self.key_var.set(os.urandom(32).hex())
            if not self.plain_path_var.get():
                raise ValueError("请先选择明文文件")
            out_path = encrypt_file(self.key_var.get(), Path(self.plain_path_var.get()))
            messagebox.showinfo("成功", f"已加密到: {out_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加密失败", str(exc))

    def _decrypt_handler(self) -> None:
        try:
            if not self.cipher_path_var.get():
                raise ValueError("请先选择密文文件")
            out_path = decrypt_file(self.key_var.get(), Path(self.cipher_path_var.get()))
            messagebox.showinfo("成功", f"已解密到: {out_path}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("解密失败", str(exc))

    def _encrypt_text_handler(self) -> None:
        try:
            if not self.key_var.get().strip():
                self.key_var.set(os.urandom(32).hex())
            plaintext = self.plain_text.get("1.0", tk.END).encode("utf-8")
            if not plaintext.strip():
                raise ValueError("请输入明文")
            blob = encrypt_bytes(self.key_var.get(), plaintext)
            if self.mode_var.get() == "phrase":
                profile_name = self.phrase_choice.get()
                if profile_name not in self.phrase_profiles:
                    raise ValueError("请选择短语表")
                value_to_choices, _, _ = self.phrase_profiles[profile_name]
                b64_text = base64.b64encode(blob).decode("ascii")
                encoded = _phrase_encode_from_base64(b64_text, value_to_choices)
            else:
                alphabet, pad = self.alphabets[self.alphabet_choice.get()]
                encoded = _encode_custom(blob, alphabet, pad)
            self.cipher_text.delete("1.0", tk.END)
            self.cipher_text.insert(tk.END, encoded)
            messagebox.showinfo("成功", "文本已加密")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加密失败", str(exc))

    def _decrypt_text_handler(self) -> None:
        try:
            cipher_b64 = self.cipher_text.get("1.0", tk.END).strip()
            if not cipher_b64:
                raise ValueError("请输入密文")
            if self.mode_var.get() == "phrase":
                profile_name = self.phrase_choice.get()
                if profile_name not in self.phrase_profiles:
                    raise ValueError("请选择短语表")
                _, phrase_to_value, prefix_set = self.phrase_profiles[profile_name]
                b64_text = _phrase_decode_to_base64(cipher_b64, phrase_to_value, prefix_set)
                blob = base64.b64decode(b64_text)
            else:
                alphabet, pad = self.alphabets[self.alphabet_choice.get()]
                blob = _decode_custom(cipher_b64, alphabet, pad)
            plaintext = decrypt_bytes(self.key_var.get(), blob)
            self.plain_text.delete("1.0", tk.END)
            self.plain_text.insert(tk.END, plaintext.decode("utf-8"))
            messagebox.showinfo("成功", "文本已解密")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("解密失败", str(exc))


def main() -> None:
    root = tk.Tk()
    AESApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
