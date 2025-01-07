#import thu vien can thiet
import numpy as np 
import torch.nn as nn
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#tao modal
def MBA(X_train, y_train, X_test, y_test):
    #chuan hoa du lieu
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #chuyen du lieu sang tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    #tao modal
    model = nn.Sequential(
        nn.Linear(3, 10),
        nn.ReLU(),
        nn.Linear(10, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    #chon ham loss va thuat toan toi uu
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    #train modal
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    #kiem tra do chinh xac
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = (y_pred > 0.5).float()
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy: {accuracy}')
    return model, sc

#chia du lieu thanh 2 phan train va test
def data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

#tao du lieu mau voi diem xet hang co cong thuc la diem xet hang =  diem hanh vi + diem hoc luc + diemtb*2
def data_create():
    np.random.seed(42)
    behavior_score = np.random.randint(0, 5, 1000)
    academic_score = np.random.randint(0, 11, 1000)
    average_score = np.random.randint(0, 11, 1000)
    ranking_score = average_score * 2 + behavior_score + academic_score
    X = np.column_stack((behavior_score, academic_score, average_score))
    y = (ranking_score >= 24).astype(int)
    return X, y

def main():
    X, y = data_create()
    X_train, X_test, y_train, y_test = data(X, y)
    model, sc = MBA(X_train, y_train, X_test, y_test)
    
    # luu model
    torch.save(model, 'model.pth')
    # tai model
    model = torch.load('model.pth')
    
    #thu nghiem du lieu voi thuat toan da dua ra
    X_new = np.array([[1, 5, 8]])
    X_new = sc.transform(X_new)
    X_new = torch.tensor(X_new, dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_new)
        y_pred = (y_pred > 0.5).float()
        print(y_pred)
        
    #in ra diem ban dau va diem du doan xem sinh vien do co dat diem hay khong
    print(f'X_new: {X_new}')
    print(f'y_pred: {y_pred}')

if __name__ == '__main__':
    main()