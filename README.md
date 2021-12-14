# Interpreting-MDETR-Using-Attribution-Guided-Factorization

This repository contains our implementation of Attribution Guided Factorization, a model interpretability method, applied upon a state-of-the-art query-modulated object detection model MDETR.

The code should be able to run out of the box in Google's Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RU-CS535-ADOX/Interpreting-MDETR-Using-Attribution-Guided-Factorization/blob/main/demo.ipynb).

## Building Blocks

The implementation contains the following two building blocks:

### MDETR: Modulated Detection for End-to-End Multi-Modal Understanding

[Website](https://ashkamath.github.io/mdetr_page/) • [Paper](https://arxiv.org/abs/2104.12763)

### MDETR Model Architecture

```
MDETR(
  (transformer): Transformer(
    (encoder): TransformerEncoder(
      (layers): ModuleList(
        (0): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (1): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (2): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (3): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (4): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
        (5): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
      )
    )
    (decoder): TransformerDecoder(
      (layers): ModuleList(
        (0): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (cross_attn_image): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
          (dropout4): Dropout(p=0.1, inplace=False)
        )
        (1): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (cross_attn_image): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
          (dropout4): Dropout(p=0.1, inplace=False)
        )
        (2): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (cross_attn_image): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
          (dropout4): Dropout(p=0.1, inplace=False)
        )
        (3): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (cross_attn_image): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
          (dropout4): Dropout(p=0.1, inplace=False)
        )
        (4): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (cross_attn_image): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
          (dropout4): Dropout(p=0.1, inplace=False)
        )
        (5): TransformerDecoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (cross_attn_image): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=2048, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=2048, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm4): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout3): Dropout(p=0.1, inplace=False)
          (dropout4): Dropout(p=0.1, inplace=False)
        )
      )
      (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (text_encoder): RobertaModel(
      (embeddings): RobertaEmbeddings(
        (word_embeddings): Embedding(50265, 768, padding_idx=1)
        (position_embeddings): Embedding(514, 768, padding_idx=1)
        (token_type_embeddings): Embedding(1, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): RobertaEncoder(
        (layer): ModuleList(
          (0): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (1): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (2): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (3): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (4): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (5): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (6): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (7): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (8): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (9): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (10): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (11): RobertaLayer(
            (attention): RobertaAttention(
              (self): RobertaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): RobertaSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): RobertaIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
            )
            (output): RobertaOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (pooler): RobertaPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
    )
    (resizer): FeatureResizer(
      (fc): Linear(in_features=768, out_features=256, bias=True)
      (layer_norm): LayerNorm((256,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (class_embed): Linear(in_features=256, out_features=256, bias=True)
  (bbox_embed): MLP(
    (layers): ModuleList(
      (0): Linear(in_features=256, out_features=256, bias=True)
      (1): Linear(in_features=256, out_features=256, bias=True)
      (2): Linear(in_features=256, out_features=4, bias=True)
    )
  )
  (query_embed): Embedding(100, 256)
  (input_proj): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
  (backbone): Joiner(
    (0): TimmBackbone(
      (body): EfficientNetFeatures(
        (conv_stem): Conv2dSame(3, 48, kernel_size=(3, 3), stride=(2, 2), bias=False)
        (bn1): FrozenBatchNorm2d()
        (act1): SiLU(inplace=True)
        (blocks): Sequential(
          (0): Sequential(
            (0): DepthwiseSeparableConv(
              (conv_dw): Conv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48, bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(48, 12, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(12, 48, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pw): Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): Identity()
            )
            (1): DepthwiseSeparableConv(
              (conv_dw): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(24, 6, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(6, 24, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pw): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): Identity()
            )
            (2): DepthwiseSeparableConv(
              (conv_dw): Conv2d(24, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=24, bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(24, 6, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(6, 24, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pw): Conv2d(24, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): Identity()
            )
          )
          (1): Sequential(
            (0): InvertedResidual(
              (conv_pw): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2dSame(144, 144, kernel_size=(3, 3), stride=(2, 2), groups=144, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(144, 6, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(6, 144, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (1): InvertedResidual(
              (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (2): InvertedResidual(
              (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (3): InvertedResidual(
              (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (4): InvertedResidual(
              (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(240, 240, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=240, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
          )
          (2): Sequential(
            (0): InvertedResidual(
              (conv_pw): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2dSame(240, 240, kernel_size=(5, 5), stride=(2, 2), groups=240, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(240, 10, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(240, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (1): InvertedResidual(
              (conv_pw): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(384, 16, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (2): InvertedResidual(
              (conv_pw): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(384, 16, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (3): InvertedResidual(
              (conv_pw): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(384, 16, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (4): InvertedResidual(
              (conv_pw): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(384, 384, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=384, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(384, 16, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
          )
          (3): Sequential(
            (0): InvertedResidual(
              (conv_pw): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2dSame(384, 384, kernel_size=(3, 3), stride=(2, 2), groups=384, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(384, 16, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(16, 384, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(384, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (1): InvertedResidual(
              (conv_pw): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (2): InvertedResidual(
              (conv_pw): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (3): InvertedResidual(
              (conv_pw): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (4): InvertedResidual(
              (conv_pw): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (5): InvertedResidual(
              (conv_pw): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (6): InvertedResidual(
              (conv_pw): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
          )
          (4): Sequential(
            (0): InvertedResidual(
              (conv_pw): Conv2d(128, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(768, 768, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=768, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(768, 32, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(32, 768, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(768, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (1): InvertedResidual(
              (conv_pw): Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1056, 1056, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1056, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (2): InvertedResidual(
              (conv_pw): Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1056, 1056, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1056, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (3): InvertedResidual(
              (conv_pw): Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1056, 1056, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1056, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (4): InvertedResidual(
              (conv_pw): Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1056, 1056, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1056, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (5): InvertedResidual(
              (conv_pw): Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1056, 1056, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1056, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (6): InvertedResidual(
              (conv_pw): Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1056, 1056, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1056, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
          )
          (5): Sequential(
            (0): InvertedResidual(
              (conv_pw): Conv2d(176, 1056, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2dSame(1056, 1056, kernel_size=(5, 5), stride=(2, 2), groups=1056, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1056, 44, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1056, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (1): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (2): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (3): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (4): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (5): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (6): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (7): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (8): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
          )
          (6): Sequential(
            (0): InvertedResidual(
              (conv_pw): Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(1824, 1824, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1824, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(1824, 76, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(1824, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (1): InvertedResidual(
              (conv_pw): Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(3072, 3072, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3072, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(3072, 128, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
            (2): InvertedResidual(
              (conv_pw): Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn1): FrozenBatchNorm2d()
              (act1): SiLU(inplace=True)
              (conv_dw): Conv2d(3072, 3072, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=3072, bias=False)
              (bn2): FrozenBatchNorm2d()
              (act2): SiLU(inplace=True)
              (se): SqueezeExcite(
                (conv_reduce): Conv2d(3072, 128, kernel_size=(1, 1), stride=(1, 1))
                (act1): SiLU(inplace=True)
                (conv_expand): Conv2d(128, 3072, kernel_size=(1, 1), stride=(1, 1))
                (gate): Sigmoid()
              )
              (conv_pwl): Conv2d(3072, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn3): FrozenBatchNorm2d()
            )
          )
        )
      )
    )
    (1): PositionEmbeddingSine()
  )
  (contrastive_align_projection_image): Linear(in_features=256, out_features=64, bias=True)
  (contrastive_align_projection_text): Linear(in_features=256, out_features=64, bias=True)
)
```

#### Model checkpoint paths

The model checkpoint for MDETR and dataset annotations can be found at: <https://zenodo.org/record/4721981#.YawS29DMKUk>.

- ResNet-101: <https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth>
- EfficientNet-B3: <https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth>
- EfficientNet-B5: <https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth>
- CLEVR: <https://zenodo.org/record/4721981/files/clevr_checkpoint.pth>
- CLEVR-Humans: <https://zenodo.org/record/4721981/files/clevr_humans_checkpoint.pth>
- ResNet-101-GQA: <https://zenodo.org/record/4721981/files/gqa_resnet101_checkpoint.pth>
- EfficientNet-B5-GQA: <https://zenodo.org/record/4721981/files/gqa_EB5_checkpoint.pth>
- ResNet-101-PhraseCut: <https://zenodo.org/record/4721981/files/phrasecut_resnet101_checkpoint.pth>
- EfficientNet-B3-PhraseCut: <https://zenodo.org/record/4721981/files/phrasecut_EB3_checkpoint.pth>
- ResNet-101-RefCOCO: <https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth>
- EfficientNet-B3-RefCOCO: <https://zenodo.org/record/4721981/files/refcoco_EB3_checkpoint.pth>
- ResNet-101-RefCOCO+: <https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth>
- EfficientNet-B3-RefCOCO+: <https://zenodo.org/record/4721981/files/refcoco%2B_EB3_checkpoint.pth>
- ResNet-101-RefCOCOg: <https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth>
- EfficientNet-B3-RefCOCOg: <https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth>

### AGF: Attribution Guided Factorization

[Conference Recording](https://slideslive.com/38949126/visualization-of-supervised-and-selfsupervised-neural-networks-via-attribution-guided-factorization) • [Paper](https://arxiv.org/abs/2012.02166)
