from internal_model import Model

M = Model()
M.parse_wnids('wnids.txt')
M.load_train_data()