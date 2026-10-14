# Lab 2: A neural network from scratch

## 1. Objective and dataset

Implement a neural network with two ReLU hidden layers using NumPy, derive its gradients, validate them numerically, and compare hidden widths and learning rates. Scikit-learn is used only to load and split the dataset; forward propagation, loss, backpropagation, and SGD are implemented directly.

The `load_digits` dataset contains 1,797 images of handwritten digits, each represented by 64 pixel values. Inputs are divided by 16 to obtain values in [0, 1]. Labels are one-hot encoded into 10 classes.

The stratified 70/30 split uses `random_state=42`: **1,257 training examples** and **540 test examples**. The test set is evaluated after training and does not participate in parameter updates.

## 2. Network design

Both hidden layers have width $W$, with experiments for $W=16$ and $W=64$:

```text
64 inputs -> W units -> ReLU -> W units -> ReLU -> 10 outputs -> softmax
```

| Parameter | General shape | W = 16 | W = 64 |
|---|---|---|---|
| $W_1$ | $(64,W)$ | $(64,16)$ | $(64,64)$ |
| $b_1$ | $(W,)$ | $(16,)$ | $(64,)$ |
| $W_2$ | $(W,W)$ | $(16,16)$ | $(64,64)$ |
| $b_2$ | $(W,)$ | $(16,)$ | $(64,)$ |
| $W_3$ | $(W,10)$ | $(16,10)$ | $(64,10)$ |
| $b_3$ | $(10,)$ | $(10,)$ | $(10,)$ |

The parameter count is $W^2+76W+10$: **1,482** for W=16 and **8,970** for W=64. Weights are sampled independently from a zero-mean normal distribution with standard deviation **0.01**. Biases start at zero.

## 3. Forward propagation and loss

For a mini-batch $X$ with shape $(m,64)$:

$$
Z_1=XW_1+b_1, \qquad A_1=\max(0,Z_1)
$$

$$
Z_2=A_1W_2+b_2, \qquad A_2=\max(0,Z_2)
$$

$$
Z_3=A_2W_3+b_3
$$

Bias vectors broadcast across batch rows. For each row, subtract the largest logit to obtain $S$. Stable log-probabilities and probabilities are

$$
\log P_{ik}=S_{ik}-\log\left(\sum_j e^{S_{ij}}\right), \qquad P_{ik}=e^{\log P_{ik}}.
$$

The output has shape $(m,10)$. Every row must sum to 1 within $10^{-6}$. Mean cross-entropy is

$$
L=-\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{10}Y_{ik}\log P_{ik}.
$$

Using log-softmax avoids taking the logarithm of an underflowed probability. Nearly uniform initial probabilities should give $L\approx\ln(10)=2.302585$.

| Width | Initial full-training loss | Maximum row-sum error on a 32-example batch |
|---|---|---|
| 16 | 2.30258552 | 2.220e-16 |
| 64 | 2.30259249 | 4.441e-16 |

Both output-shape and probability-sum assertions passed, and both initial losses were within 0.01 of ln(10).

## 4. Backpropagation derivation

For one sample, cross-entropy gives $\partial L/\partial P_k=-Y_k/P_k$. The softmax derivative is

$$
\frac{\partial P_k}{\partial Z_j}=P_k(\delta_{kj}-P_j).
$$

Applying the chain rule and using $\sum_kY_k=1$ gives

$$
\frac{\partial L}{\partial Z_j}=\sum_k\frac{-Y_k}{P_k}P_k(\delta_{kj}-P_j)=P_j-Y_j.
$$

For the mean loss over a batch, define $D_3=(P-Y)/m$. The remaining gradients follow from the matrix-product chain rule:

$$
\nabla W_3=A_2^TD_3, \qquad \nabla b_3=\sum_iD_{3,i}
$$

$$
D_2=(D_3W_3^T)\odot\mathbf{1}[Z_2>0]
$$

$$
\nabla W_2=A_1^TD_2, \qquad \nabla b_2=\sum_iD_{2,i}
$$

$$
D_1=(D_2W_2^T)\odot\mathbf{1}[Z_1>0]
$$

$$
\nabla W_1=X^TD_1, \qquad \nabla b_1=\sum_iD_{1,i}.
$$

The elementwise masks are the ReLU derivatives. At zero, the implementation uses derivative 0. Bias gradients sum over the batch because the same bias is added to every sample. Each parameter gradient has the same shape as its parameter.

### Numerical validation

Each width is checked using its trained learning-rate-0.1 parameters and the first 32 training examples. A trained snapshot avoids cancellation associated with extremely small initial derivatives. Ten weight entries are sampled uniformly without replacement, using seed 2026; entries are not filtered or resampled.

For each selected weight, use central differences with $\epsilon=10^{-5}$, then restore the original value:

$$
g_n=\frac{L(\theta+\epsilon)-L(\theta-\epsilon)}{2\epsilon}, \qquad
r=\frac{|g_n-g_a|}{\max(10^{-12},|g_n|+|g_a|)}.
$$

At least 9 of 10 entries must satisfy $r<10^{-6}$ for each width. Zero analytical and numerical slopes count as passing under this definition.

| Width | Passing weights | Maximum relative error | ReLU sign crossings in perturbed passes |
|---|---|---|---|
| 16 | 10/10 | 2.835e-09 | 0 |
| 64 | 10/10 | 4.662e-08 | 0 |

Both widths pass the required criterion. Individual analytical slopes, numerical slopes, and relative errors are recorded in [gradient_check.csv](results/gradient_check.csv).

## 5. Training experiments

Training uses plain mini-batch SGD:

$$
\theta\leftarrow\theta-\eta\nabla_\theta L.
$$

| Setting | Value |
|---|---|
| Hidden widths | 16, 64 |
| Learning rates | 0.001, 0.01, 0.1 |
| Epochs | 50 per configuration |
| Batch size | 32, including the final smaller batch |
| Initialization seed | 42 |
| Shuffle seed | 123, reset for each run |
| Arithmetic | float64; one BLAS thread |

Training examples are shuffled once per epoch. For each width, all three learning rates share the same initialization and mini-batch order. Logged epoch loss and training accuracy are computed on the full training set after the final update of the epoch. All 300 epoch records are available in [epoch_logs.csv](results/epoch_logs.csv).

### Final results after 50 epochs

| Width | Learning rate | Training loss | Training accuracy | Test accuracy | Correct test predictions |
|---|---|---|---|---|---|
| 16 | 0.001 | 2.302550 | 10.10% | 10.19% | 55/540 |
| 16 | 0.01 | 2.302479 | 10.10% | 10.19% | 55/540 |
| 16 | 0.1 | 2.407343 | 24.90% | 23.89% | 129/540 |
| 64 | 0.001 | 2.302526 | 10.10% | 10.19% | 55/540 |
| 64 | 0.01 | 2.302344 | 10.10% | 10.19% | 55/540 |
| 64 | 0.1 | 0.188314 | 93.16% | 90.74% | 490/540 |

### Loss curves

![Mean training cross-entropy over 50 epochs](results/loss_curves.png)

With learning rates 0.001 and 0.01, both widths remain near uniform-prediction loss and reach only 10.19% test accuracy within 50 epochs. Small initial weights and two ReLU layers produce small early hidden-layer gradients, which is consistent with this slow optimization.

With learning rate 0.1, the wider model begins improving earlier and reaches **90.74% test accuracy (490/540)**. The W=16 model improves late, but its loss spikes near the end and finishes at 2.407343 with 23.89% test accuracy, indicating unstable training in this configuration.

The highest observed accuracy occurs at W=64 and learning rate 0.1. These comparisons use the prescribed test split and one initialization per configuration; they do not establish a statistically reliable ranking or constitute validation-based hyperparameter selection.

## 6. Forward and backward timing

Timing uses a batch of 32 training examples, trained learning-rate-0.1 parameters, 20 warm-up iterations, and the **median of 100 runs** per pass. Measurement order alternates to reduce order bias. Backward uses an existing forward cache; its timing excludes computing that cache. Data loading, updates, and the explicit loss calculation are excluded. Forward includes stable log-softmax and probability calculation.

| Width | Median forward (ms) | Median backward (ms) | Backward / forward |
|---|---|---|---|
| 16 | 0.028600 | 0.028700 | 1.003 |
| 64 | 0.043250 | 0.054050 | 1.250 |

For W=16, forward and backward are essentially tied; the small difference does not establish meaningful dominance. For W=64, backward takes about **25% longer**.

Forward performs three matrix multiplications. Backward performs five: three products for weight gradients and two for propagating errors into hidden layers. An input gradient is not needed. For batch size $m$, the multiply-accumulate counts are

$$
\text{forward}=m(64W+W^2+10W), \qquad
\text{backward}=m(64W+2W^2+20W).
$$

| Width | Forward multiply-accumulates, m=32 | Backward multiply-accumulates, m=32 |
|---|---|---|
| 16 | 46,080 | 59,392 |
| 64 | 282,624 | 434,176 |

The hidden-to-hidden matrix product grows quadratically with width, increasing the relative cost of backward propagation for a wider network. Softmax, masks, reductions, memory allocation, and function-call overhead also affect measured time, so runtime ratios need not equal arithmetic-count ratios. Results are machine-dependent.

## 7. Conclusion and reproduction

The NumPy network passes output and loss sanity checks, and numerical differentiation validates the implemented backpropagation for both widths. Of the six prescribed runs, W=64 with learning rate 0.1 achieves the highest observed final test accuracy, 90.74%. The low-rate runs show that 50 epochs with small initial weights can yield little optimization progress. Timing shows that backward propagation is more expensive for the wider network, consistent with its additional matrix multiplications.

Run from the `lab02` directory:

```bash
python -m pip install -r requirements.txt
python lab2.py
```

The run writes the loss plot and result tables to `results/`. Jupyter is not required for the standalone Python implementation. Experiment settings and software versions are recorded in [results.json](results/results.json); final metrics and timings are also available as [summary.csv](results/summary.csv) and [timing.csv](results/timing.csv).
