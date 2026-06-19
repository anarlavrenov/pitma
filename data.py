import torch


class IterableFlushDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, max_seq_len, eos_id, pad_id):
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.eos_id = eos_id
        self.pad_id = pad_id

    def _emit(self, window_tok, window_doc):

        pad_n = self.max_seq_len - len(window_tok)
        if pad_n > 0:
            window_tok = window_tok + [self.pad_id] * pad_n
            window_doc = window_doc + [0] * pad_n

        return (
            torch.tensor(window_tok, dtype=torch.long),
            torch.tensor(window_doc, dtype=torch.long)
        )

    def __iter__(self):

        window_tokens = []
        window_doc_ids = []
        doc_id = 0

        for example in self.dataset:
            if not example:
                continue

            tokens = self.tokenizer.encode(example, add_special_tokens=False) + [self.eos_id]

            # Если документ сам по себе > max_seq_len.
            if len(tokens) > self.max_seq_len:

                # Если сложилась ситуация, где документ > max_seq_len, но
                # окно уже содержит токены предыдущих документов, просто еще не
                # заполнено до конца, то заполняю паддингами текущее окно и выдаю его.
                if window_tokens:
                    yield self._emit(window_tokens, window_doc_ids)
                    window_tokens, window_doc_ids, doc_id = [], [], 0

                for s in range(0, len(tokens), self.max_seq_len):
                    chunk = tokens[s: s + self.max_seq_len]
                    # Если чанк = max_seq_len, то его одного и выдаю.
                    if len(chunk) == self.max_seq_len:
                        yield self._emit(chunk, [1] * self.max_seq_len)

                    # Последний чанк (< max_seq_len) кладется в окно.
                    else:
                        window_tokens = chunk
                        window_doc_ids = [1] * len(chunk)
                        doc_id = 1

                continue

            # Если документ не влезает в текущее окно (оно уже содержит токены),
            # то выдаю текущее окно как есть, текущий документ не кладу.
            if len(window_tokens) + len(tokens) > self.max_seq_len:
                yield self._emit(window_tokens, window_doc_ids)
                window_tokens, window_doc_ids, doc_id = [], [], 0

            doc_id += 1
            window_tokens.extend(tokens)
            window_doc_ids.extend([doc_id] * len(tokens))

        if window_tokens:
            yield self._emit(window_tokens, window_doc_ids)


class IterableBaseDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer):
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer

    def __iter__(self):
        for example in self.dataset:
            if not example:
                continue

            # size: (1, L)
            tokens = self.tokenizer.encode(example, return_tensors="pt", add_special_tokens=False)
            # (L)
            tokens = tokens.squeeze(0)
            tokens = tokens.to(torch.long)

            yield tokens


class PackedCollateTST:
    def __init__(self, max_seq_len, batch_size, eos_id):

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.eos_id = eos_id
        self.buffer = []

    def __call__(self, batch):

        for doc in batch:
            if doc.numel() == 0:
                continue

            self.buffer.extend(doc.tolist())
            self.buffer.append(self.eos_id)

        # Сколько окон по max_seq_len токенов собралось.
        total_windows = len(self.buffer) // self.max_seq_len
        if total_windows == 0:
            return None

        # Проверка на то, не собралось ли на данный момент окон больше,
        # чем размер batch_size.
        total_windows = min(total_windows, self.batch_size)
        # Собранное количество токенов на данный момент.
        take_n_tokens = total_windows * self.max_seq_len
        # Беру эти токены из буффера.
        taken_tokens = self.buffer[:take_n_tokens]
        # Удаляю из буффера взятые данные.
        self.buffer = self.buffer[take_n_tokens:]

        chunk = torch.tensor(taken_tokens, dtype=torch.long)
        chunk = chunk.view(total_windows, self.max_seq_len)

        src = chunk[:, :-1].contiguous()
        tgt = chunk[:, 1:].contiguous()

        # Все тензоры: size: (B, max_seq_len - 1).
        return src, tgt


def collate_fn_flush_ntp(batch):

  tokens = torch.stack([b[0] for b in batch])
  doc_ids = torch.stack([b[1] for b in batch])

  src = tokens[:, :-1].contiguous()
  tgt = tokens[:, 1:].contiguous()
  src_doc_ids = doc_ids[:, :-1].contiguous()
  tgt_doc_ids = doc_ids[:, 1:].contiguous()

  loss_mask = (src_doc_ids == tgt_doc_ids) & (tgt_doc_ids != 0)

  return src, tgt, src_doc_ids, loss_mask

