
import torch

class Config(object):
    def __init__(self, dataset):
        self.train_path = dataset + 'train.txt'     #train
        self.dev_path = dataset + 'dev.txt'         #validation
        self.test_path = dataset + 'test.txt'       #test
        # self.class_path = dataset + 'class.txt'
        self.predict_path = dataset + '/saved_data/' + 'predict.csv'    #prediction result
        self.value_path = dataset + '/saved_data/' + 'value.csv'
        self.save_path = dataset + '/saved_model/' + 'model.ckpl'

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.k_fold = 10
        self.epochs = 20
        self.batch_size = 16
        self.max_seq = 400
        self.lr = 1e-5
        self.require_improvement = 2

        # self.class_list = [x.strip() for x in open(self.class_path, encoding='utf-8').readlines()]
        # self.num_classes = len(self.class_list)
        # self.id2class = dict(enumerate(self.class_list))
        # self.class2id = {j: i for i, j in self.id2class.items()}    

        self.class_list = None
        self.num_classes = None
        self.id2class = None
        self.class2id = None

        self.num_filters = 128
        self.embed_dim = 200
