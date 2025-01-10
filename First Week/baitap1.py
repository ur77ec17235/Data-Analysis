#import thư viện cần thiết
import numpy as np
import sklearn.datasets as datasets
import sklearn.model_selection as model_selection
import sklearn.preprocessing as preprocessing
import torch

import torch.optim as optim
import torch.nn as nn


# Tạo class MLP để định nghĩa mô hình mạng nơ-ron
class MLP(nn.Module):

    def __init__(self, *args, **kwargs):
        super(MLP, self).__init__(*args, **kwargs)
        self.fc1  = nn.Linear(3, 100) # 4 là số chiều của dữ liệu đầu vào, 100 là số nơ-ron ở tầng ẩn
        self.act1 = nn.ReLU()
        self.fc2  = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.fc3  = nn.Linear(100, 1) # 3 là số lớp của dữ liệu đầu ra
        self.act3 = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        return x
    
# Tạo hàm train để huấn luyện mô hình
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(x)
            y_pred = y_pred.squeeze(1)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, 
                                                                         num_epochs, i + 1, len(train_loader), loss.item()))

#kiểm tra tính cân bằng của dữ liệu mẫu được tạo để tets mô hình
def check_balance(labels):
    n_positive = (labels == 1).sum().item()
    n_negative = (labels == 0).sum().item()
    # print('Mẫu đạt:', n_positive)
    # print('Mẫu không đạt:', n_negative)
    print('Tỷ lệ mẫu đạt:', n_positive / len(labels))
    print('Tỷ lệ mẫu không đạt', n_negative / len(labels))



def test(model, test_loader):
    #chuẩn hóa dữ liệu mô hình


    model.eval() # dùng để báo cho mô hình biết là nó đang 
    # ở tình trạng kiểm tra nên không cần tính đạo hàm
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in test_loader:
            y_pred = model(x)
            y_pred = (y_pred > 0.5).float()
            total += y.size(0)
            correct += (y_pred == y).sum().item()
        print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))

# Tạo dữ liệu mẫu
def generate_sample_data(n_samples):
    np.random.seed(42)
    
    # Tạo dữ liệu ngẫu nhiên trong phạm vi hợp lý
    avg_scores = np.random.uniform(0, 10, n_samples)  # Điểm trung bình từ 0-10
    behavior_scores = np.random.randint(0, 4, n_samples)  # Điểm hạnh kiểm 0-3
    training_scores = np.random.uniform(0, 10, n_samples)  # Điểm rèn luyện 0-10
    
    # Tính điểm danh hiệu theo công thức
    honor_scores = avg_scores * 2 + behavior_scores + training_scores
    
    # Tạo nhãn (0: không đạt, 1: đạt)
    labels = (honor_scores >= 24)
    
    check_balance(labels)
   
    # Gộp features
    features = np.column_stack([avg_scores, behavior_scores, training_scores])
    

    return features, labels


def build():
    # Tạo dữ liệu mẫu
    features, labels = generate_sample_data(10000)

    # for i in range (20):
    #     print(features[i], labels[i])

    # Chia dữ liệu thành tập huấn luyện và tập kiểm tra
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Chuyển dữ liệu sang tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Tạo DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Tạo mô hình
    model = MLP()

    # Định nghĩa hàm loss và optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) #lr laf learning rate

    # Huấn luyện mô hình
    train(model, train_loader, criterion, optimizer, num_epochs=10)

    # Kiểm tra mô hình
    test(model, test_loader)

    # Lưu mô hình
    torch.save(model.state_dict(), 'model.pth')

def main():
    # # dữ liệu ví dụ cho mô hình được tạo theo hình thức thủ công
    # hs1 = [6.5, 2, 9]
    # hs2 = [5, 3, 7]
    # hs3 = [7, 1, 8]
    # hs4 = [9, 0, 10]
    # hs5 = [10, 3, 10]
    # hs6 = [6, 2, 7]
    #áp dụng mô hình để đnáh giá điểm danh hiệu cảu học sinh
    # nếu điểm danh hiệu >= 24 thì học sinh đó đạt

    # Tạo mô hình từ dữ liệu mẫu
    build()
    
    # tạo dữ liệu mẫu
    features, labels = generate_sample_data(12)
    # for i in range (12):
    #     print(features[i], labels[i])


    # Tạo mô hình từ file đã lưu
    model = MLP()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    # Chuẩn hóa dữ liệu
    scaler = preprocessing.StandardScaler()
    features = scaler.fit_transform(features)
    features = torch.tensor(features, dtype=torch.float32)

    # Dự đoán
    with torch.no_grad():
        y_pred = model(features)
        y_pred = (y_pred > 0.5).float()
        print('Nhãn dự đoán:', y_pred)
        print('Nhãn thực tế:', labels)

    # in ra các học sinh đạt dnah hiệu 
    # và các học sinh không đạt danh hiệu với số điểm tương ứng mà sinh viên đó có
    for i in range(12):
        original_scores = scaler.inverse_transform(features[i].numpy().reshape(1, -1)).flatten()
        print(f"Inverse transformed features for student {i+1}: {original_scores}")
        if y_pred[i] == 1:
            print(f'Học sinh {i+1} đạt danh hiệu với điểm:', original_scores)
        else:
            print('Học sinh không đạt danh hiệu với điểm:', original_scores)

if __name__ == '__main__':
    main()

