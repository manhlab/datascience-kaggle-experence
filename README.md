Intermediate Machine Learning
```
https://www.kaggle.com/learn/intermediate-machine-learning
```
100 colab NLP LAB
```
https://notebooks.quantumstat.com/
```
Colab config
```
https://towardsdatascience.com/colab-free-gpu-ssh-visual-studio-code-server-36fe1d3c5243
```
Colab
```
https://colab.research.google.com/drive/1fhU2hTMf-jj7WllmaWLSJiBllXnRJPwf#scrollTo=wGeght7jBFTF
```
JS to autoconnect
```
%%javascript

function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
}
setInterval(ConnectButton,60000);
```
Label Smoothing
```
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)
            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = self.confidence * nll_loss + self.smoothing * smooth_loss
            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)
```
KFOLD 

```
class LabelSmoothedCrossEntropyLoss(nn.Module):
    """this loss performs label smoothing to compute cross-entropy with soft labels, when smoothing=0.0, this
    is the same as torch.nn.CrossEntropyLoss"""

    def __init__(self, n_classes, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_classes = n_classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
  ```
 Pseudo-Labeling to deal with small datasets 
 ```
    https://towardsdatascience.com/pseudo-labeling-to-deal-with-small-datasets-what-why-how-fd6f903213af
 ```
Class Balance TPU
```
https://www.kaggle.com/shonenkov/class-balance-with-pytorch-xla
```
