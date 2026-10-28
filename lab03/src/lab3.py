"""Run from lab03: python src/lab3.py --data data --threads 4"""
import argparse, json, platform, statistics, time
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
from thop import profile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(49, 64)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(64, 4, 128, batch_first=True) for _ in range(2)])
        self.head = nn.Linear(64, 10)
    def forward(self, x):
        x = x.unfold(2,7,7).unfold(3,7,7).permute(0,2,3,1,4,5).reshape(x.size(0),16,49)
        x = self.projection(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(1))

def count_attention(m, inputs, output):
    x = inputs[0]; b,s,d = x.shape
    m.total_ops += torch.DoubleTensor([b*(4*s*d*d+2*s*s*d)])

def accuracy(model, loader, device):
    model.eval(); correct=total=0
    with torch.inference_mode():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            correct += (model(x).argmax(1)==y).sum().item(); total += len(y)
    return correct/total

def latency(model, batch, runs, device):
    x=torch.randn(batch,1,28,28,device=device); model.eval()
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(20): model(x)
        sync(); times=[]
        for _ in range(runs):
            sync(); start=time.perf_counter(); model(x); sync()
            times.append((time.perf_counter()-start)*1000)
    return statistics.median(times)

def main():
    lab_dir=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(); p.add_argument('--data',default=str(lab_dir/'data')); p.add_argument('--threads',type=int,default=4)
    args=p.parse_args(); out=lab_dir/'results'; out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42); torch.set_num_threads(args.threads)
    # Disable fused encoder fast paths so THOP sees the same explicit modules.
    torch.backends.mha.set_fastpath_enabled(False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2860,),(0.3530,))])
    full=datasets.FashionMNIST(args.data,train=True,download=True,transform=transform)
    train,val=random_split(full,[55000,5000],generator=torch.Generator().manual_seed(42))
    test=datasets.FashionMNIST(args.data,train=False,download=True,transform=transform)
    trainloader=DataLoader(train,batch_size=128,shuffle=True,generator=torch.Generator().manual_seed(42))
    valloader=DataLoader(val,batch_size=128); testloader=DataLoader(test,batch_size=128)
    models={
      'MLP':nn.Sequential(nn.Flatten(),nn.Linear(784,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU(),nn.Linear(128,10)),
      'CNN':nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,10)),
      'Attention':Attention()}
    results=[]; history=[]
    analytical={'MLP':469504,'CNN':2063488,'Attention':2329856}
    for name,model in models.items():
        model.to(device)
        (out/f'{name}_summary.txt').write_text(str(summary(model,input_size=(1,1,28,28),device=device,verbose=0)),encoding='utf-8')
        model.eval(); sample=torch.randn(1,1,28,28,device=device)
        macs,_=profile(model,inputs=(sample,),verbose=False)
        corrected,_=profile(model,inputs=(sample,),custom_ops={nn.MultiheadAttention:count_attention},verbose=False)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
        for epoch in range(1,6):
            model.train(); loss_sum=0
            for x,y in trainloader:
                x,y=x.to(device),y.to(device); optimizer.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(model(x),y); loss.backward(); optimizer.step(); loss_sum+=loss.item()*len(y)
            va=accuracy(model,valloader,device)
            history.append(dict(model=name,epoch=epoch,train_loss=loss_sum/len(train),validation_accuracy=va))
            print(name,epoch,'validation',round(va,4),flush=True)
        acc=accuracy(model,testloader,device)
        row=dict(model=name,parameters=sum(v.numel() for v in model.parameters()),analytical_flops=analytical[name],thop_raw_macs=macs,thop_attention_hook_macs=corrected,test_accuracy=acc,latency_b1_ms=latency(model,1,100,device),latency_b128_ms=latency(model,128,20,device))
        results.append(row); torch.save(model.state_dict(),out/f'{name}.pt')
        (out/'results.json').write_text(json.dumps(results,indent=2))
        print(row,flush=True)
    (out/'history.json').write_text(json.dumps(history,indent=2))
    for key,label,filename in [('parameters','Parameters','accuracy_vs_parameters.png'),('latency_b1_ms','Batch 1 latency (ms)','accuracy_vs_latency.png')]:
        fig,ax=plt.subplots(figsize=(7,4.5))
        for r in results:
            ax.scatter(r[key],100*r['test_accuracy'],s=65); ax.annotate(r['model'],(r[key],100*r['test_accuracy']),xytext=(6,6),textcoords='offset points')
        ax.set(xlabel=label,ylabel='Test accuracy (%)',title='Fashion-MNIST: five epochs'); ax.grid(alpha=.25); ax.margins(.2); fig.tight_layout(); fig.savefig(out/filename,dpi=180); plt.close(fig)
    rows='\n'.join(f"| {r['model']} | {r['parameters']:,} | {100*r['test_accuracy']:.2f}% | {r['latency_b1_ms']:.3f} | {r['latency_b128_ms']:.3f} |" for r in results)
    costs='\n'.join(f"| {r['model']} | {r['analytical_flops']:,} | {r['thop_raw_macs']:,.0f} | {2*r['thop_attention_hook_macs']:,.0f} |" for r in results)
    winner=max(results,key=lambda r:r['test_accuracy'])['model']
    fastest=min(results,key=lambda r:r['latency_b1_ms'])['model']
    smallest=min(results,key=lambda r:r['parameters'])['model']
    report=f'''# Lab 3: Architectures and their costs

## Protocol
Fashion-MNIST official 60,000 training images split into 55,000 train and 5,000 validation with seed 42; official 10,000 test images. ToTensor and Normalize((0.2860,), (0.3530,)). All models: Adam, learning rate 0.001, batch 128, five epochs. Test evaluated once after the final epoch; validation is recorded each epoch. Device: {device}; Python {platform.python_version()}, PyTorch {torch.__version__}; CPU: {platform.processor()}; threads: {args.threads}. No claim of bitwise reproducibility across platforms. No positional embeddings are added because the specification does not request them; this limits spatial information in the attention model.

## Architecture discrepancy
The exact specified MLP has 235,146 parameters, CNN 20,490, and attention 70,794. Thus the requested 100k-150k size target conflicts with the explicit layer definitions. This experiment preserves the explicit definitions. Parameter count is therefore a confounder, and this is not a controlled comparison at equal size. Attention uses PyTorch encoder defaults (ReLU, dropout 0.1, post-normalization, biases).

## Results
| Model | Parameters | Test accuracy | Batch 1 median ms | Batch 128 median ms |
|---|---:|---:|---:|---:|
{rows}

Latency excludes data loading and host/device transfer. Synthetic tensors already reside on the device. Evaluation mode and inference_mode; 20 warm-up runs per batch size, then median of 100 runs for batch 1 and 20 for batch 128. CUDA synchronization brackets timings when relevant. Batch latency is not per-image latency; divide batch-128 time by 128 for amortized cost. These are local measurements, not hardware-independent properties.

## Hand calculation
Multiply and add count as two FLOPs; biases, activations, pooling, normalization, and softmax are excluded from the analytical matrix-operation baseline.

* MLP first Linear: 2*784*256 = 401,408 FLOPs. Parameters: 784*256+256 = 200,960. Whole model: 2*(784*256+256*128+128*10) = 469,504.
* CNN first convolution: 2*3*3*1*16*28*28 = 225,792 FLOPs. Parameters: 3*3*1*16+16 = 160. Second convolution: 2*3*3*16*32*14*14 = 1,806,336. Classifier: 2*1568*10 = 31,360. Total = 2,063,488.
* Attention sequence length 16, width 64, four heads of width 16. QKV per encoder: 2*16*64*3*64 = 393,216; QK^T: 2*16^2*64 = 32,768; attention-times-V: 32,768; output projection: 2*16*64*64 = 131,072. Attention subtotal = 589,824. Feed-forward: 2*16*(64*128+128*64) = 524,288. Two encoders = 2,228,224. Patch projection = 2*16*49*64 = 100,352; head = 1,280. Whole model = 2,329,856 FLOPs.

### Handwritten working

![Handwritten MLP and CNN parameter and FLOP calculations](evidence/hand_calculations.jpg)

The supplied scan records the MLP and CNN working. The typed derivations above also include attention and give the values used in the profiling comparison.

## Profiling comparison
| Model | Analytical matrix FLOPs | Raw THOP MACs | Hook-adjusted THOP x2 |
|---|---:|---:|---:|
{costs}

THOP primarily reports MACs rather than two-operation FLOPs. Raw THOP can omit MultiheadAttention functional operations. The custom hook counts QKV, output projection, scores, and weighted values. LayerNorm and other operations may be counted with tool-specific conventions, so doubling all reported operations is only an approximation. torchinfo layer summaries are saved separately. Analytical FLOPs exclude non-matrix operations, and parameter storage alone does not predict latency.

## Trade-offs and access patterns
![Accuracy against parameters](results/accuracy_vs_parameters.png)

![Accuracy against latency](results/accuracy_vs_latency.png)

Highest observed test accuracy: {winner}. Fastest batch-1 inference: {fastest}; fewest parameters: {smallest}. For this run, CNN offers the strongest accuracy/storage trade-off, while MLP is preferable when local inference latency is the priority. Attention does not justify its extra complexity at this patch size and five-epoch training budget.

The CNN reuses small kernels over local neighborhoods and shares weights, reducing parameter storage but creating feature-map traffic. Pooling reduces downstream activation sizes. The MLP streams a larger dense weight matrix; batch 1 offers limited weight reuse, while batching reuses weights across images and improves matrix multiplication efficiency. Attention projects tokens with dense operations and forms score matrices; score storage grows quadratically with token count. At only 16 tokens, launch/dispatch, normalization, softmax, and small matrix operations can dominate. Batching improves utilization but increases activation memory. These access-pattern interpretations follow the lab's Chapter 4.6-4.7 topics; the chapter text was not supplied, so no direct quotations are claimed.

Cost-effectiveness depends on the constraint: use the table and plots to compare accuracy, parameter memory, and measured latency together. A higher-FLOP model can still run faster because FLOPs omit memory traffic and execution overhead. Five epochs and one seed provide an initial comparison, not statistical proof of architectural superiority.

## Reproduce
From `lab03`, install requirements.txt, then run `python src/lab3.py --data PATH_TO_DATA --threads 4`. Results, training history, plots, summaries, and final model weights are written in `lab03/results/`; the report is written to `lab03/report.md`. These output paths are independent of the working directory. Without `--data`, the dataset cache defaults to `lab03/data/`. The first run downloads Fashion-MNIST.
'''
    (out.parent/'report.md').write_text(report,encoding='utf-8')

if __name__=='__main__': main()

