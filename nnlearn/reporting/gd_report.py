from nnlearn.metrics import accuracy_score

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from rich.table import Table
from rich.columns import Columns
from rich.panel import Panel
from rich.markdown import Markdown

class GdReport:
    """
    Report for the models optimized via gradient descent.
    """

    def __init__(self, figpath):
        self.figpath = figpath

        self.table = None
        self.fig = None
        self.report = None
        self.epochs_mean_losses = []
        self.epochs_accuraccies = []

        self._setup_report_table()

    def _setup_report_table(self):
        
        self.table = table = Table()
        self.table.add_column("Epoch", justify="center", style="subtle")
        self.table.add_column("Loss", justify="center", style="rose")
        self.table.add_column("Accuracy", justify="center", style="love")

    def eval_epoch(self, epoch, losses, y_train, yhat_train):

        # Mean loss
        mean_loss = np.mean(losses)
        
        # Accuracy
        acc = accuracy_score(y_train, yhat_train)

        # Save them for plotting
        self.epochs_mean_losses.append(mean_loss)
        self.epochs_accuraccies.append(acc)

        # Create table row
        epoch = "{:05d}".format(epoch)
        mean_loss = "{:06.3f}".format(mean_loss)
        acc = "{:04.2f}".format(acc)
        self.table.add_row(epoch, mean_loss, acc)

    def create_report(self, loss_func_name, batch_size, lr):
        
        # Define plot
        self.fig, axs = plt.subplots(2, 1, sharex=True)
        plt.tight_layout();
        self.fig.subplots_adjust(wspace=0.2, hspace=.2);

        # Define epochs
        n_epochs = len(self.epochs_mean_losses)
        epochs = np.arange(1, n_epochs + 1, 1)

        # Epochs vs loss
        sns.lineplot(x=epochs, y=self.epochs_mean_losses, ax=axs[0], label='loss')
        axs[0].set_title("Epoch # vs loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        
        # Epochs vs accuracy
        sns.lineplot(x=epochs, y=self.epochs_accuraccies, ax=axs[1], label='accuracy') 
        axs[1].set_title("Epoch # vs accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        
        # Save the first figure
        fig1path = f"{self.figpath}training.png"
        self.fig.savefig(fig1path, bbox_inches='tight')
        
        # Create markdown
        fig1path_md = "/".join(fig1path.split("/")[1:])
        md = ""
        md += "## Hyper-parameters\n"
        md += "Following hyper-parameters have been used:\n"
        md += f"- Epochs: {n_epochs}\n"
        md += f"- Loss func: {loss_func_name}\n"
        md += f"- Batch size: {batch_size}\n"
        md += f"- LR: {lr}\n"
        md += "## Training plot\n"
        md += f"📈 See training plot [here]({fig1path_md})\n"

        panel_1 = Panel.fit(self.table, title="table of training", width=35)
        panel_2 = Panel.fit(Markdown(md), title="training information", width=35)
        self.report = Columns([panel_1, panel_2])

