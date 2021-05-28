from transformers import PreTrainedModel, PreTrainedTokenizer
import torch
from typing import Optional
from torch.utils.data.dataloader import DataLoader


class SummarizerTrainer:
        def __init__(self,
                     model: PreTrainedModel,
                     tokenizer: PreTrainedTokenizer,
                     optimizer: torch.optim.Optimizer,
                     n_epochs: int,
                     device: str = 'cpu'):         
            """
            Complete training and evaluation loop in Pytorch specially designed for
            Transformer-based models from transformers library (HuggingFace).

            Args:
                model (`PreTrainedModel`): Pre-trained model from transformers library.
                tokenizer (`PreTrainedTokenizer`): Pre-trained tokenizer from transformers library.
                optimizer (`torch.optim.Optimizer`): Pytorch Optimizer
                n_epochs (`int`): Number of epochs to train.
                device (`str`): Type of device where to train the network. It must be `cpu` or `cuda`.

            Methods:
                train(dataloader_train: DataLoader, dataloader_val: Optional[DataLoader] = None)
                    Complete training and evaluation (optional) loop in Pytorch.
                predict(article: str, max_article_len: int = 512, max_summary_len: int = 150)
                    Generates summary for a given article (string).

            """
            self.tokenizer = tokenizer
            self.model = model
            self.optimizer = optimizer
            self.n_epochs = n_epochs
            self.device = device

               
        def train(self,
                  dataloader_train: DataLoader,
                  dataloader_val: Optional[DataLoader] = None):
            """
            Complete training and evaluation (optional) loop in Pytorch.
            Args:
                dataloader_train (`torch.utils.data.dataloader.DataLoader`): Pytorch dataloader.
                dataloader_val (`torch.utils.data.dataloader.DataLoader`, `Optional`):
                    Pytorch dataloader. If `None` no validation will be performed.

            """
            self.model.to(self.device)
            train_len = len(dataloader_train)
            val_len = len(dataloader_val)
            
            for epoch in range(1, self.n_epochs + 1):
                # Training
                # -----------------------------
                self.model.train()
                tr_loss_cum = .0
                self.model.train()
                for i, batch in enumerate(dataloader_train, 1):
                    # Send batch to device, estimate the error, acumulate the error,
                    # Estimate and print the mean error over the batches, reset the old gradient, 
                    # estimate the new gradient and update model weights.
                    ###     YOUR CODE HERE      ###
                    batch = {key:torch.squeeze(value, dim=1).to(self.device) for key,value in batch.items()}
                    # Labels with -100 aren't considered to calculate the loss and we dont want to calculate the loss
                    # on the padded tokens
                    batch['labels'][batch['labels']==self.tokenizer.pad_token_id] = -100 

                    self.optimizer.zero_grad()
                    output = self.model(**batch)
                    output.loss.backward()
                    self.optimizer.step()

                    tr_loss_cum += output.loss.item()

                # Validation
                # -----------------------------
                if dataloader_val is not None:
                    self.model.eval()
                    val_loss_cum = .0

                    for i, batch in enumerate(dataloader_val, 1):
                        # Iterate over the batches from the `dataloader_val`, send batch to device,
                        # estimate the validation error, accumulate the error, and estimate and print 
                        # the validation error (mean of the error over all batches)
                        ###     YOUR CODE HERE      ###
                        batch = {key:torch.squeeze(value, dim=1).to(self.device) for key,value in batch.items()}
                        batch['labels'][batch['labels']==self.tokenizer.pad_token_id] = -100
                        with torch.no_grad():
                            output = self.model(**batch)
                            val_loss_cum += output.loss.item()
                
                print(f"Epoch {epoch} -> Loss: (Train) {tr_loss_cum/train_len} (Validation) {val_loss_cum/val_len}")
                    

        def predict(self, 
                    article: str,
                    max_article_len: int = 512,
                    max_summary_len: int = 150) -> str:

            """
            Generates a summary for a given input article.
            Args:
                article (`str`): String that contains the article.
                max_article_len (`int`): Maximum length sequence for the article.
                max_summary_len (`int`): Maximum length sequence for the summary.
                    
            Returns:
                summary_pred (`str`): String that represents the predicted summary.

            """
            # Transform input text into tensors (T5 inputs)
            ###     YOUR CODE HERE      ###
            article = self.tokenizer('summarize: ' + article, return_tensors='pt', padding='max_length', max_length=max_article_len, truncation=True)
            article.to(self.device)
            self.model.eval()
            # Generate ids for the given input
            with torch.no_grad():
                ###     YOUR CODE HERE      ###
                # generated = self.model.generate(**article, max_lenght = max_summary_len, early_stopping=True, num_beams=2)
                generated = self.model.generate(input_ids = article['input_ids'],
                                                attention_mask = article['attention_mask'],
                                                max_length=max_summary_len,
                                                early_stopping=True,
                                                num_beams=2)
            # Transform generated ids into text
            ###     YOUR CODE HERE      ###
            summary_pred = self.tokenizer.batch_decode(generated,
                                                       skip_special_tokens=True,
                                                       clean_up_tokenization_spaces=True)
            return summary_pred

                
                
