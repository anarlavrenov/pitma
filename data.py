import torch


class IterableDataWrapper(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer

    def __iter__(self):
        for example in self.dataset:

            if not example:
                continue

            # size: (1, L)
            tokens = self.tokenizer.encode(example, return_tensors="pt")
            # (L)
            tokens = tokens.squeeze(0)
            tokens = tokens.to(torch.long)

            yield tokens


class PackedCollate:
    def __init__(self, max_seq_len, batch_size, eos_id):

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.eos_id = eos_id
        self.buffer = []
        self.doc_ids = []

        self.doc_id = 0

    def __call__(self, batch):

        for doc in batch:
            if doc.numel() == 0:
                continue

            self.buffer.extend(doc.tolist())
            self.buffer.append(self.eos_id)

            self.doc_ids.extend([self.doc_id + 1] * (doc.shape[0] + 1))
            self.doc_id += 1

        # Сколько окон по 4096 токенов собралось.
        total_windows = len(self.buffer) // self.max_seq_len
        if total_windows == 0:
            empty_data = torch.empty((0, self.max_seq_len - 1), dtype=torch.long)
            empty_doc = torch.empty((0, self.max_seq_len - 1), dtype=torch.long)
            empty_mask = torch.empty((0, self.max_seq_len - 1), dtype=torch.bool)

            return empty_data, empty_data, empty_doc, empty_mask

        # Проверка на то, не собралось ли на данный момент окон больше,
        # чем размер batch_size.
        total_windows = min(total_windows, self.batch_size)
        # Собранное количество токенов на данный момент.
        take_n_tokens = total_windows * self.max_seq_len
        # Беру эти токены из буффера.
        taken_tokens = self.buffer[:take_n_tokens]
        taken_doc_ids = self.doc_ids[:take_n_tokens]
        # Удаляю из буффера взятые данные.
        self.buffer = self.buffer[take_n_tokens:]
        self.doc_ids = self.doc_ids[take_n_tokens:]

        if self.doc_ids:
            # Хвост. 16
            offset = self.doc_ids[0]
            # [0, 0, 0, 0]
            self.doc_ids = [d - offset for d in self.doc_ids]
            # 0
            self.doc_id = self.doc_ids[-1]
        else:
            self.doc_id = 0

        chunk = torch.tensor(taken_tokens, dtype=torch.long)
        chunk = chunk.view(total_windows, self.max_seq_len)

        chunk_doc_ids = torch.tensor(taken_doc_ids, dtype=torch.long)
        chunk_doc_ids = chunk_doc_ids.view(total_windows, self.max_seq_len)

        src = chunk[:, :-1].contiguous()
        tgt = chunk[:, 1:].contiguous()

        src_doc_ids = chunk_doc_ids[:, :-1].contiguous()
        tgt_doc_ids = chunk_doc_ids[:, 1:].contiguous()

        # Токены, которые находятся на пересечении документов будут игнорироваться.
        loss_mask = src_doc_ids == tgt_doc_ids

        # Все тензоры: size: (B, max_seq_len - 1).
        return src, tgt, src_doc_ids, loss_mask