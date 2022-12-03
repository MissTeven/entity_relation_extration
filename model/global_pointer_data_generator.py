from bert4keras.snippets import sequence_padding, DataGenerator


class GlobalPointerDataGenerator(DataGenerator):
    def __init__(self, data, batch_size, tokenizer, max_token_length, predicate2id):
        super().__init__(data, batch_size)
        self.tokenizer = tokenizer
        self.predicate2id = predicate2id
        self.max_token_length = max_token_length

        """数据生成器
        """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_entity_labels, batch_head_labels, batch_tail_labels, batch_head_entity_labels, batch_tail_entity_labels = \
            [], [], [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = self.tokenizer.encode(d['text'], maxlen=self.max_token_length)
            # 整理三元组 {(s, o, p)}
            spoes = set()
            for s, p, o in d['spo_list']:
                s = self.tokenizer.encode(s)[0][1:-1]
                p = self.predicate2id[p]
                o = self.tokenizer.encode(o)[0][1:-1]
                sh = self.search(s, token_ids)
                oh = self.search(o, token_ids)
                if sh != -1 and oh != -1:
                    spoes.add((sh, sh + len(s) - 1, p, oh, oh + len(o) - 1))
            # 构建标签
            entity_labels = [set() for _ in range(2)]
            head_entity_labels = [set() for _ in range(3)]
            tail_entity_labels = [set() for _ in range(3)]
            head_labels = [set() for _ in range(len(self.predicate2id))]
            tail_labels = [set() for _ in range(len(self.predicate2id))]
            for sh, st, p, oh, ot in spoes:
                entity_labels[0].add((sh, st))
                entity_labels[1].add((oh, ot))

                head_entity_labels[p_head_map.get(p)].add((sh, st))
                tail_entity_labels[p_tail_map.get(p)].add((oh, ot))

                head_labels[p].add((sh, oh))
                tail_labels[p].add((st, ot))
            for label in entity_labels + head_labels + tail_labels + head_entity_labels + tail_entity_labels:
                if not label:  # 至少要有一个标签
                    label.add((0, 0))  # 如果没有则用0填充
            entity_labels = sequence_padding([list(l) for l in entity_labels])
            head_entity_labels = sequence_padding([list(l) for l in head_entity_labels])
            tail_entity_labels = sequence_padding([list(l) for l in tail_entity_labels])
            head_labels = sequence_padding([list(l) for l in head_labels])
            tail_labels = sequence_padding([list(l) for l in tail_labels])
            # 构建batch
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_entity_labels.append(entity_labels)
            batch_head_entity_labels.append(head_entity_labels)
            batch_tail_entity_labels.append(tail_entity_labels)
            batch_head_labels.append(head_labels)
            batch_tail_labels.append(tail_labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_entity_labels = sequence_padding(
                    batch_entity_labels, seq_dims=2
                )
                batch_head_entity_labels = sequence_padding(
                    batch_head_entity_labels, seq_dims=2
                )
                batch_tail_entity_labels = sequence_padding(
                    batch_tail_entity_labels, seq_dims=2
                )
                batch_head_labels = sequence_padding(
                    batch_head_labels, seq_dims=2
                )
                batch_tail_labels = sequence_padding(
                    batch_tail_labels, seq_dims=2
                )
                yield [batch_token_ids, batch_segment_ids], [
                    batch_entity_labels, batch_head_labels, batch_tail_labels, batch_head_entity_labels,
                    batch_tail_entity_labels
                ]
                batch_token_ids, batch_segment_ids = [], []
                batch_entity_labels, batch_head_labels, batch_tail_labels, batch_head_entity_labels, \
                batch_tail_entity_labels = [], [], [], [], []

    def search(self, pattern, sequence):
        """从sequence中寻找子串pattern
        如果找到，返回第一个下标；否则返回-1。
        """
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1
