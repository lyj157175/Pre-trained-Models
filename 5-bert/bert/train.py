from torch.optim import Adam
from torch.utils.data import DataLoader
from vocab import Args, vocab, train_data_loader, test_data_loader
from model import BERT
import torch.nn as nn
import torch
import tqdm



class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with two LM training method.
        1. Masked Language Model 
        2. Next Sentence prediction 
    """
    def __init__(self, 
                bert: BERT, 
                vocab_size: int,
                train_dataloader: DataLoader, 
                test_dataloader: DataLoader = None,
                lr: float = 1e-4, 
                betas=(0.9, 0.999), 
                weight_decay: float = 0.01,
                with_cuda: bool = True, 
                log_freq: int = 10): 

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.bert = bert
        self.model = BERTLM(bert, vocab_size).to(self.device)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.criterion = nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"

        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            next_sent_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # loss of next_sentence_prediction
            next_loss = self.criterion(next_sent_output, data["is_next"])
            # loss of masked LM
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])
            loss = next_loss + mask_loss

            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            # next sentence prediction accuracy
            correct = next_sent_output.argmax(dim=-1).eq(data["is_next"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_next"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/bert_trained.model"):
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.bert.cpu(), output_path)
        self.bert.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path



class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """
    def __init__(self, bert: BERT, vocab_size):

        super().__init__()
        self.bert = bert
        self.next_sentence = NextSentencePrediction(self.bert.hidden)
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.next_sentence(x), self.mask_lm(x)


class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """
    def __init__(self, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))



bert = BERT(len(vocab), hidden=256, n_layers=8, attn_heads=8)
args = Args()
trainer = BERTTrainer(bert, 
                    len(vocab), 
                    train_dataloader=train_data_loader, 
                    test_dataloader=test_data_loader,
                    lr=args.lr, 
                    betas=(args.adam_beta1, args.adam_beta2), 
                    weight_decay=args.adam_weight_decay,
                    with_cuda=args.with_cuda, 
                    log_freq=args.log_freq)

for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch, args.output_path) 

    if test_data_loader is not None:
        trainer.test(epoch)