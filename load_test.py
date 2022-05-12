from utils import load_txt_data, load_cite_data

# 测试读取.txt文件格式数据

adj, features, labels, idx_train, idx_val, idx_test = load_txt_data()
# adj, features, labels, idx_train, idx_val, idx_test = load_cite_data()
print(features.shape)