import numpy as np
import os
from tqdm import tqdm

class RunnerM():
    """
    This is an exmaple to train, evaluate, save, load the model. However, some of the function calling may not be correct 
    due to the different implementation of those models.
    """
    def __init__(self, model, optimizer, metric, loss_fn, batch_size=32, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric = metric
        self.scheduler = scheduler
        self.batch_size = batch_size

        self.train_scores = []
        self.dev_scores = []
        self.train_loss = []
        self.dev_loss = []

    def train(self, train_set, dev_set, **kwargs):

        num_epochs = kwargs.get("num_epochs", 0)
        log_iters = kwargs.get("log_iters", 100)
        save_dir = kwargs.get("save_dir", "best_model")

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        best_score = 0

        epoch_pbar = tqdm(range(num_epochs), desc='Training', unit='epoch')
        for epoch in epoch_pbar:
            X, y = train_set

            assert X.shape[0] == y.shape[0]
            
            idx = np.random.permutation(range(X.shape[0]))
            
            X = X[idx]
            y = y[idx]

            
            batch_pbar = tqdm(range(int(X.shape[0] / self.batch_size) + 1), 
                            desc=f'Epoch {epoch}', 
                            leave=False,
                            unit='batch')
            
            for iteration in batch_pbar:
                # 在每次迭代前清零梯度
                self.optimizer.zero_grad()
                train_X = X[iteration * self.batch_size : (iteration+1) * self.batch_size]
                train_y = y[iteration * self.batch_size : (iteration+1) * self.batch_size]

                logits = self.model(train_X)
                trn_loss = self.loss_fn(logits, train_y)
                self.train_loss.append(trn_loss)
                
                trn_score = self.metric(logits, train_y)
                self.train_scores.append(trn_score)

                # the loss_fn layer will propagate the gradients.
                self.loss_fn.backward()

                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                
                dev_score, dev_loss = self.evaluate(dev_set)
                self.dev_scores.append(dev_score)
                self.dev_loss.append(dev_loss)

                if (iteration) % log_iters == 0:
                    print(f"epoch: {epoch}, iteration: {iteration}")
                    print(f"[Train] loss: {trn_loss}, score: {trn_score:.7f}")  # 增加小数位数
                    print(f"[Dev] loss: {dev_loss}, score: {dev_score:.7f}")
                    print(f"参数更新检查: ")
                    for i, layer in enumerate(self.model.layers[:2]):
                        if layer.optimizable:
                            print(f"Layer {i} W max: {layer.params['W'].max()}, min: {layer.params['W'].min()}")
                batch_pbar.set_postfix({
                    'train_loss': f'{trn_loss:.4f}',
                    'train_acc': f'{trn_score:.4f}',
                    'val_acc': f'{dev_score:.4f}'
                })
            epoch_pbar.set_postfix({
                'best_val_acc': f'{best_score:.4f}',
                'current_val_acc': f'{dev_score:.4f}'
            })    

            if dev_score > best_score:
                save_path = os.path.join(save_dir, 'best_model.pickle')
                self.save_model(save_path)
                print(f"best accuracy performence has been updated: {best_score:.5f} --> {dev_score:.5f}")
                best_score = dev_score
        self.best_score = best_score

    def evaluate(self, data_set):
        X, y = data_set
        logits = self.model(X)
        loss = self.loss_fn(logits, y)
        score = self.metric(logits, y)
        return score, loss
    
    def save_model(self, save_path):
        self.model.save_model(save_path)