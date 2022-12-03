from bert4keras.layers import GlobalPointer
from bert4keras.models import build_transformer_model
from bert4keras.backend import keras, K
from keras.models import Model


class GlobalPointerWrap(Model):
    def __init__(self, config_path, checkpoint_path, predicate2id):
        # 加载预训练模型
        base = build_transformer_model(
            config_path=config_path,
            model='nezha',
            checkpoint_path=checkpoint_path,
            return_keras_model=False
        )
        print(base.layers)

        base_model_output = keras.layers.average([base.layers["Transformer-23-FeedForward-Norm"].output,
                                                  base.layers["Transformer-22-FeedForward-Norm"].output,
                                                  base.layers["Transformer-21-FeedForward-Norm"].output,
                                                  base.layers["Transformer-20-FeedForward-Norm"].output])

        base_model_output = keras.layers.Dropout(0.3)(base_model_output)
        # 预测结果
        entity_output = GlobalPointer(heads=2, head_size=64)(base_model_output)
        head_entity_output = GlobalPointer(heads=3, head_size=64)(base_model_output)
        tail_entity_output = GlobalPointer(heads=3, head_size=64)(base_model_output)
        head_output = GlobalPointer(
            heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
        )(base_model_output)
        tail_output = GlobalPointer(
            heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
        )(base_model_output)

        outputs = [entity_output, head_output, tail_output, head_entity_output, tail_entity_output]

        super.__init__(base.model.inputs, outputs)
