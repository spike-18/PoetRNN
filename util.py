import torch.nn as nn



class RModule(nn.Module):
    """
    Базовый класс посимвольной рекуррентной нейросети
    input_size: число ключей словаря. Испольщует encoder для кодирования токе-вектор
    output_size: число ключей словаря
    hidden_size: длина вектора скрытого состояния
    embedding_size: размер входного вектора
    n_layers: число слоев рекуррентной сети
    encoder: nn.Embedding переводит токен в вектор длиной embedding_size
    decoder: nn.Linear - переводит вектор скрытого состояния в логиты
    rnn: nn.RNN / nn.LSTM - должен быть определен классом-наследником
    """
    def __init__(self, input_size, hidden_size, embedding_size=18, n_layers=1):
        super(RModule, self).__init__()
        self.input_size = input_size
        self.output_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, embedding_size)
        self.decoder = nn.Linear(hidden_size, self.output_size)
        self.rnn = None
    
    def forward(self, input, hidden):
        input = self.encoder(input)
        output, hidden = self.rnn(input, hidden)
        output = self.decoder(output)
        return output, hidden