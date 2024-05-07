# Step-by-step Guid for Adding a New Method

`GCondenser` is a unified graph condenser toolkit that covers everything from data preprocessing to condensed graph evaluation. Adding a new graph condenser to `GCondenser` is easy and effortless. Test your graph condensation methods with the current state-of-the-art in GC.

## Step One
Add a new file named `your_graph_condenser.py` in the `./graph_condenser/models/` folder.

```python
import dgl
from graph_condenser.models.condenser import Condenser


class YourGraphCondenser(Condenser):
    def __init__(
        self,
        dataset: dgl.data.DGLDataset,
        observe_mode: str,
        budget: int,
        label_distribution: str,
        init_method: str,
        val_mode: str,
        gnn_val: torch.nn.Module,
        lr_val: float,
        wd_val: float,
        gnn_test: torch.nn.Module,
        lr_test: float,
        wd_test: float,
        ...
        lr_feat: float,  # for example, this is a hyperparameter of your method.
    ):
        super().__init__(
            dataset,
            observe_mode,
            budget,
            label_distribution,
            init_method,
            val_mode,
            gnn_val,
            lr_val,
            wd_val,
            gnn_test,
            lr_test,
            wd_test,
        )
    
    def training_step(self, data, batch_idx):
        ```
        Define how your graph condenser to update the condenser graph.
        ```
        # Your codes here
    
    def configure_optimizers(self):
        ```
        Define the optimizer for updating the condensed graph.
        ```
        # Your codes here
```

## Step Two
Create a config file for pass nesscessary parameters for your method. For example, create a file `./configs/condenser/your_graph_condenser.yaml`:
```yaml
_target_: graph_condenser.models.your_graph_condenser.YourGraphCondenser

observe_mode: transductive
budget: 30
label_distribution: original # original, balanced, strict
init_method: kCenter # kCenter, randomChoice

val_mode: GNN
gnn_val:
  _target_: graph_condenser.models.backbones.gcn.GCN
  _partial_: true
  dropout: 0.5
lr_val: 1e-2
wd_val: 5e-4

gnn_test:
  _target_: graph_condenser.models.backbones.gcn.GCN
  _partial_: true
  dropout: 0.5
lr_test: 1e-2
wd_test: 5e-4

lr_feat: 1e-4  # this is your graph condenser's hyperparameter
```

## Step Three
Run the following commad in your terminal.
```bash
python graph_condenser/train.py condenser=your_graph_condener
```
