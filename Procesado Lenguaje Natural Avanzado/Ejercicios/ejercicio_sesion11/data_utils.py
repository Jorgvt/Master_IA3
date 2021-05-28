import torch
from typing import Tuple, Dict
import pandas as pd
from transformers import PreTrainedTokenizer


class SummarizerDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer, article_len, summary_len):
        """
        Class that builds a torch Dataset specially designed for Summarization data.
        Args:
            dataset (`pandas.DataFrame`): Pandas DataFrame that contains two columns:
                `ctext` (article) and `text` (summary).
            tokenizer (`PreTrainedTokenizer`): Pre-trained tokenizer from transformers
                For T5 is loaded as `T5Tokenizer.from_pretrained(...)`.
            article_len (`int`): Maximum length sequence for the article.
            summary_len (`int`): Maximum length sequence for the summary.

        """
        super(SummarizerDataset).__init__()
        self.tokenizer = tokenizer
        self.data = dataset
        self.article_len = article_len
        self.summary_len = summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return text2tensors(article=str(self.data.ctext[i]), 
                            summary=str(self.data.text[i]), 
                            tokenizer=self.tokenizer, 
                            article_length=self.article_len, 
                            summary_length=self.summary_len)


def load_data(path: str, 
              tr_ratio: float = 0.8,
              seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the News Summary Dataset (https://www.kaggle.com/sunnysai12345/news-summary)
    and process it to be used by T5. It mainly removes NaN values and concatenates the 
    str 'summarize: ' to each summary. Finally, it suffles the data and split it into
    training and validation.
    Args:
        path (`str`): Path where the data is stored.
        tr_ratio (`float`): ratio of training data.
        seed (`int`): State of the random function. It is used for reproducibility
        purposes.
        summary_len (`int`): Maximum length sequence for the summary.
    
    Returns:
        training_set (`pandas.DataFrame`): Training data.
        val_set (`pandas.DataFrame`): Validation data.

    """
    df = pd.read_csv(path, encoding='latin-1')[['text','ctext']]
    df = df.dropna()
    df.ctext = 'summarize: ' + df.ctext
    training_set = df.sample(frac=tr_ratio, random_state = seed)
    val_set = df.drop(training_set.index).reset_index(drop=True)
    training_set = training_set.reset_index(drop=True)
    return training_set, val_set


def text2tensors(article: str, 
                 summary: str, 
                 tokenizer: PreTrainedTokenizer,
                 article_length: int,
                 summary_length: int) -> Dict[str, torch.tensor]:
    """
    Takes data (article and summary) and converts it into tensors to feed the 
    neural network (T5).
    Args:
        article (`str`): String that contains the article.
        summary (`str`): String that contains the summary (target)
        tokenizer (`PreTrainedTokenizer`): Pre-trained tokenizer from transformers
            library. For T5 is loaded as `T5Tokenizer.from_pretrained(...)`.
        article_length (`int`): Maximum length sequence for the article.
        summary_length (`int`): Maximum length sequence for the summary.

    Returns:
        t5_input (`dict`): Dictionary where the keys are the arguments that T5 needs
        as input. Values are the corresponding Torch tensors. 
        {'input_ids': torch.Tensor, 
         'attention_mask': torch.Tensor, 
         'decoder_input_ids': torch.Tensor, 
         'labels': torch.Tensor}

    """
    article = ' '.join(article.split())
    summary = ' '.join(summary.split())

    # Convert article and summary into tensors
    ###     YOUR CODE HERE      ###
    # article = torch.Tensor(article)
    # summary = torch.Tensor(summary)

    # Build input of the encoder (ids and attention mask), input of the decoder
    # and process the labels. Then return the dictionary with all the generated data.
    ###     YOUR CODE HERE      ###
    article_tokenized = tokenizer(article, return_tensors='pt',return_attention_mask=True, padding='max_length', max_length=article_length, truncation=True)
    summary_tokenized = tokenizer(summary, return_tensors='pt',return_attention_mask=True, padding='max_length', max_length=summary_length, truncation=True)
    labels = summary_tokenized.input_ids[:,1:]

    t5_input = {
        'input_ids':article_tokenized.input_ids,
        'attention_mask':article_tokenized.attention_mask,
        'decoder_input_ids':summary_tokenized.input_ids[:,:-1],
        'labels':labels
    }

    return t5_input