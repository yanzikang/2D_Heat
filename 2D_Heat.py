import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import torch

# 由于我使用的是pytorch所以在执行的时候需要指定框架
# CUDA_VISIBLE_DEVICES=3 DDEBACKEND=pytorch python 2D_Heat.py

# 定义计算区域
geom = dde.geometry.Rectangle([-1,-1],[1,1])
timedomain = dde.geometry.TimeDomain(0,10)
spatial_time_domain = dde.geometry.GeometryXTime(geom,timedomain)

# 定义PDE方程
alpha = 0.5
def pde(x,y):
    dy_t = dde.grad.jacobian(y,x,i=0,j=2)
    dy_xx = dde.grad.hessian(y,x,i=0,j=0)
    dy_yy = dde.grad.hessian(y,x,i=1,j=1)
    return dy_t - alpha*(dy_xx + dy_yy)

# 定义边界条件
# 用lambda表达式进行定义,提供了Dirichlet边界条件和Neumann边界条件
# 上边界，y=1
bc_top = dde.icbc.NeumannBC(
    spatial_time_domain,
    lambda x:0,
    lambda x,on_boundary:on_boundary and np.isclose(x[1],1))

# 下边界，y=-1
bc_bottom = dde.icbc.NeumannBC(
    spatial_time_domain,
    lambda x:20,
    lambda x,on_boundary:on_boundary and np.isclose(x[1],-1))  

# 左边界，x=-1
bc_left = dde.icbc.DirichletBC(
    spatial_time_domain,
    lambda x:30,
    lambda x,on_boundary:on_boundary and np.isclose(x[0],-1))

# 右边界，x=1
bc_right = dde.icbc.DirichletBC(
    spatial_time_domain,
    lambda x:50,
    lambda x,on_boundary:on_boundary and np.isclose(x[0],1))

# 定义初始条件
ic = dde.icbc.IC(
    spatial_time_domain,
    lambda x:0,
    lambda _,on_initial:on_initial)

# 汇总边界条件
ibcs = [bc_top,bc_bottom,bc_left,bc_right,ic]
# 定义数据
data = dde.data.TimePDE(
    spatial_time_domain,
    pde,
    ibcs,
    num_domain=8000,
    num_boundary=320,
    num_initial=800,
    num_test=8000,
)
# 定义网络结构
net = dde.nn.FNN([3] + 4 * [50] + [1], "tanh", "Glorot normal")
model = dde.Model(data,net)
model.compile("adam",lr=1e-3)

# 训练模型并绘制损失图
losshistory,train_state = model.train(epochs=10000,  display_every=1000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# 保存模型
net = model.net
torch.save(net, "2D_Heat_" + "pinn.pth")

# 结果绘制自己用AI写就好
#-----------------------------------------------------绘制结果--------------------------------------#
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib as plt
import os

# xy方向离散200个节点
x1 = np.linspace(-1, 1, num=500, endpoint=True).flatten()
y1 = np.linspace(-1, 1, num=500, endpoint=True).flatten()
xx1, yy1 = np.meshgrid(x1, y1)
x = xx1.flatten()
y = yy1.flatten()

# 时间上取20个时间步，时间步长1/20=0.05s
Nt = 40
dt = 1/Nt


for n in range(0, Nt + 1):
    t = n * dt
    t_list = t * np.ones([len(x), 1])  # 创建时间向量
    x_pred = np.concatenate([x[:, None], y[:, None], t_list], axis=1)  # 拼接x,y,t
    y_pred = model.predict(x_pred)  # 模型预测
    y_p = y_pred.flatten()  # 展平预测结果
    data_n = np.concatenate([x_pred, y_pred], axis=1)  # 拼接输入和预测结果
    
    if n == 0:
        data = data_n[:, :, None]  # 初始化data
    else:
        data = np.concatenate([data, data_n[:, :, None]], axis=2)  # 沿第三维拼接
    
    print(x_pred.shape, y_pred.shape)
    print(data.shape, data_n.shape)

import os
import matplotlib.pyplot as plt
import numpy as np

# 1. 创建图片保存路径
work_path = os.path.join('2DtransientRectTC')  # 修正路径拼接
is_created = os.path.exists(work_path)        # 修正函数名和变量名
if not is_created:
    os.makedirs(work_path)                    # 修正函数名
print("保存路径：" + work_path)

# 2. 计算数据范围并初始化画布
y_min = data.min(axis=(0, 2))[3]              # 修正语法：轴参数和索引
y_max = data.max(axis=(0, 2))[3]              # 修正括号匹配
fig = plt.figure(figsize=(10, 10))            # 修正函数名和参数


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

def anim_update(t_id):
    plt.clf()  # 清除当前画布
    
    # 提取时间步t_id对应的数据（修正索引和标点）
    x1_t = data[:, 0, t_id].flatten()  # x坐标
    x2_t = data[:, 1, t_id].flatten()  # y坐标
    y_p_t = data[:, 3, t_id].flatten()  # 温度值
    
    # 绘制三角化等高线图（修正函数名和参数）
    plt.tricontourf(x1_t, x2_t, y_p_t, levels=160, cmap="coolwarm")
    
    # 添加颜色条（修正Normalize和ScalarMappable用法）
    norm = Normalize(vmin=y_p_t.min(), vmax=y_p_t.max())
    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="coolwarm"), ax=plt.gca())
    
    # 设置坐标轴标签和标题（修正字符串格式）
    plt.xlabel("$x$ (m)")
    plt.ylabel("$y$ (m)")
    plt.title(f"Temperature field at t = {round(t_id*dt, 2)} s", fontsize=12)
    
    # 保存当前帧为PNG（修正路径拼接）
    plt.savefig(f"{work_path}/frame_{t_id:03d}.png")

# 创建动画（修正frames和interval参数）
anim = FuncAnimation(
    fig, 
    anim_update,
    frames=np.arange(0, data.shape[2], dtype=np.int64),
    interval=200  # 帧间隔(ms)
)
anim.save(f"{work_path}/animation.gif", writer="pillow", dpi=300)