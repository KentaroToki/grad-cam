from tqdm import tqdm


class Logger():
    def __init__(self):
        self.log = {
            'epoch': [],
            'train_loss': [],
            'test_loss': [],
            'train_accuracy': [],
            'test_accuracy': []
        }

    def record(self, epoch, train_loss, test_loss, train_accuracy, test_accuracy, is_print=True):
        self.log['epoch'].append(epoch)
        self.log['train_loss'].append(train_loss)
        self.log['test_loss'].append(test_loss)
        self.log['train_accuracy'].append(train_accuracy)
        self.log['test_accuracy'].append(test_accuracy)
        if is_print:
            self.print()

    def print(self):
        tqdm.write('epoch: {} | train_loss: {:.4f} | test_loss: {:.4f} | train_accuracy: {:.2f}% | test_accuracy: {:.2f}%'.format(
            self.log['epoch'][-1], self.log['train_loss'][-1], self.log['test_loss'][-1], self.log['train_accuracy'][-1], self.log['test_accuracy'][-1]))
