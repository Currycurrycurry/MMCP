class EncodeData:
    def __init__(self, X, Y, Z, B):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.B = B

class MixedBatchData:
    def __init__(self, data, batch_id=-1, win_flag=True, first_flag=True):
        self.batch_id = batch_id
        self.data = data
        self.size = len(data.Z)
        self.first_flag = first_flag


class CensoredBatchData:
    def __init__(self, data, batch_id=-1, win_flag=True):
        self.batch_id = batch_id
        self.data = data
        self.win_flag = win_flag
        self.size = len(data.Z)

class CensoredBatchFirstSecondData:
    def __init__(self, data, batch_id=-1, win_flag=True, first_flag=True):
        self.batch_id = batch_id
        self.data = data
        self.win_flag = win_flag
        self.size = len(data.Z)  
        self.first_flag = first_flag
        