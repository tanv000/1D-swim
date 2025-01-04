Genomic Interpreter is a formal extension of the base paper on genomic Interpretation. While the core ideas and 
references originate from the base paper, the contribution involves significant modifications, enhancements, and 
formalization of the concept.
# 1D-Swin
1D-Swin is an efficient transformer for capturing hierarchical 1-dimentional long range sequence. 1d-works with data that’s arranged in line, swin-helps the model to understand and capture the relationships and interactions between distant parts of DNA sequences,that's how it got the name 1D-Swin.
It is efficient as swin Transformer block uses a shifted window approach, which reduces the computational complexity of the Transformer.

![Architecture of 1D-Swin Transformer]<img width="1444" alt="swin" src="https://github.com/user-attachments/assets/65d88fe0-80b4-45a3-b2e0-706bea81bb36" />


### Simple explaination for working of 1D-Swin transformer
The 1D Swin Transformer chomps through long sequences by slicing them into overlapping windows, where it closely examines each piece (attention!), then gradually connects them across the whole sequence layer by layer. This efficient, multi-scale approach lets it capture both local details and big-picture patterns, making it a champion in tasks like text analysis, DNA study, and even music generation!
## Working of 1D-Swin Transformer
![Working of 1D-Swin Transformer]![image](https://github.com/user-attachments/assets/18c3a3ec-d395-42e7-8ad2-efd00a993824)

## Dissecting the 1D-Swin Transformer
![Dissecting the 1D-Swin Transformer Key Equations Explained]![image](https://github.com/user-attachments/assets/bbf6e1ed-0f90-46c1-ba4e-8fda7de5603e)


# Install
## Install via pip

```bash
pip install git+https://github.com/Zehui127/1d-swin
```

## Install from source
  ```bash
  git clone https://github.com/Zehui127/1d-swin
  cd 1d-swin
  pip install -e .
  ```

# Demo
```python
from swin1d.module import swin_1d_block
from swin1d.examples import (
    random_text_generator,
    generate_random_dna,
    onehot_encoder,
)

Base 1D-Swin Transformer code
def test_genomic_model(seq_length=512):
    """The input is a random DNA sequence generator, which generates a random
    DNA sequence with length of seq_length. The output is a tensor with shape
    of (batch_size, seq_length//block_num, hidden_size*block_num)."""

    input = generate_random_dna(seq_length)
    encode_input = onehot_encoder(input)
    model = swin1d_block(4)
    output = model(encode_input)
    print(output.shape)
    return output


def test_language_model(seq_length=512):
    """The input is a random text generator, which generates a random text with
    length of seq_length. The output is a tensor with shape of
    (batch_size, seq_length//block_num, input_token_size*block_num)."""

    input = random_text_generator(2, seq_length, tokenized=True)
    model = swin1d_block(1)
    output = model(input)
    print(output.shape)
    return output


def swin1d_block(dim):
    # stage = (number layers in each swin,
    #          whether to merge the ouput of each swin,
    #          window size)

    window_size = 32
    stages = [
        (
            4,
            True,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
        (
            2,
            False,
            window_size,
        ),
    ]
    model = swin_1d_block(stages, dim)
    return model

```

# Result
![Output1](https://github.com/user-attachments/assets/d4de551e-6af0-4467-a170-9131bc8cad5c)
![Output2](https://github.com/user-attachments/assets/18eda0e3-a97b-419e-803e-4546fc202673)


# Cite This Project

```bibtex
@article{li2023genomic,
  title={Genomic Interpreter: A Hierarchical Genomic Deep Neural Network
         with 1D Shifted Window Transformer},
  author={Li, Zehui and
          Das, Akashaditya and
          Beardall, William AV and
          Zhao, Yiren and
          Stan, Guy-Bart},
  journal={arXiv preprint arXiv:2306.05143},
  year={2023}
}
