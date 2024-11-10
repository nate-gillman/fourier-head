<div align="center">

# <img src="misc/assets/fouriers_head.png" alt="Fourier's head" width="30"/> Fourier Head:<br />Helping Large Language Models Learn<br />Complex Probability Distributions

[![arXiv](https://img.shields.io/badge/arXiv-2410.22269-<COLOR>.svg)](https://arxiv.org/abs/2410.22269)
[![Project Page](https://img.shields.io/badge/Project%20page-8A2BE2)](https://nategillman.com/fourier-head)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

The official PyTorch implementation of the paper [**"Fourier Head: Helping Large Language Models Learn Complex Probability Distributions"**](https://arxiv.org/abs/2410.22269).
Please visit our [**webpage**](https://nategillman.com/fourier-head) for more details.

![teaser](misc/assets/toy_example_gmm.gif)

## Using the Fourier head in your own work

> [!TIP]
> If you're looking for an example to cannibalize for your own experiments, we recommend you start with the [synthetic toy example](/toy-example-synthetic/README.md), since the implementation is minimalist and self-contained.

In [fourier_head.py](fourier_head.py) we implement the Fourier head.
This is a self-contained file, so you can copy it into your codebase and import it like you would any neural network layer.
Example usage:

```python
import torch
from fourier_head import Fourier_Head

class MyNetwork(torch.nn.Module):
    def __init__(self, input_dim=16, output_dim=18, num_frequencies=42, device="cpu"):
        super(MyNetwork, self).__init__()
        
        # Initialize Fourier head with specified input and output dimensions
        self.classification_head = Fourier_Head(
          input_dim, output_dim, num_frequencies, device=device
        )
        
    def forward(self, x):
        # Fourier head maps (batch_size, input_dim) --> (batch_size, output_dim)
        return self.classification_head(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
tensor_input = torch.randn(32, 16).to(device) # Batch size 32, input dimension 16
model = MyNetwork(device=device).to(device)
tensor_output = model(tensor_input)
print(tensor_output.shape)  # Expected shape: (32, 18)
```

> [!NOTE]
> * In the paper, the main use case for the Fourier head is as a drop-in replacement for the linear classification head.
Accordingly, our implementation of the Fourier head outputs the <em>inverse softmax</em> (i.e. the log) of the categorical distribution that you obtain from quantizing the learned continuous PDF. In other words: to obtain the continuous-looking Fourier head  PMFs as in the paper, you need to apply `softmax` to the output of our `Fourier_Head`.
> * Some older versions of PyTorch can't execute `torch.nn.functional.conv1d` on complex-valued tensors. We provide an implementation that works for this case inside [imitation-learning/mingpt/_fourier_head.py](imitation-learning/mingpt/_fourier_head.py).

## Recreating results from paper

Our paper contains four sets of experiments with the Fourier head.
Look inside the corresponding subdirectory for the code to recreate the results and figures from that section of the paper.

1. [Toy example (synthetic data)](/toy-example-synthetic/README.md)

2. [Toy example (audio classification)](/toy_example_audio/README.md)

3. [Large scale example (imitation learning)](/imitation-learning/README.md)

4. [Large scale example (probabilistic time series forecasting)](/time-series-forecasting/README.md)

## Acknowledgments

We thank the authors of the works we build upon:
- [SynTheory](https://huggingface.co/datasets/meganwei/syntheory)
- [Decision Transformer](https://github.com/kzl/decision-transformer)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)

## Bibtex

If you find this code useful in your research, please cite:

```
@misc{gillman2024fourierheadhelpinglarge,
  title={Fourier Head: Helping Large Language Models Learn Complex Probability Distributions}, 
  author={Nate Gillman and Daksh Aggarwal and Michael Freeman and Saurabh Singh and Chen Sun},
  year={2024},
  eprint={2410.22269},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2410.22269}, 
}
```
