class Param():
    
    def __init__(self, args):
        
        self.hyper_param = self._get_hyper_parameters(args)
        
    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """

        if args.text_backbone.startswith('bert'):

            hyper_parameters = {
                'eval_monitor': ['f1'],
                'train_batch_size': [32],
                'eval_batch_size': 8,
                'test_batch_size': 8,
                'wait_patience': [5],
                'num_train_epochs': [100],
                'base_dim': [256],
                'hidden_dim': [128],
                'lr': [1e-5],
                'warmup_proportion': [0.05], 
                'weight_hidden_dim': [768],
                'weight_dropout': [0.4],
                'weight_decay': [0.1],
                'grad_clip': [0.4],
                'gamma': [1.0],
                # loss parameter
                'temperature': [1.0],
                'powered_relevancies': [True],
                'n_samples': [32],
                'stochastic': [False],
                'beta': [0.1],
                'DEFAULT_EPS': [1e-10],
                # other
                'mean_pooling': [True],
            }            
        else:
            raise ValueError('Not supported text backbone')
        
        return hyper_parameters 
     
