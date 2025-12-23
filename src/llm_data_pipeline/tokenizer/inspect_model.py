import sentencepiece as spm

if __name__ == "__main__":
    sp = spm.SentencePieceProcessor()
    sp.Load(model_file="outputs/dev/tokenizers/spm32k/spm_32k.model")
    # 0-3: 训练时传入的特殊 token，4-259 256 个字节，
    print("piece_size =", sp.get_piece_size())  # pyright: ignore[reportAttributeAccessIssue]
    for i in range(500):
        print(i, sp.id_to_piece(i))  # pyright: ignore[reportAttributeAccessIssue]
