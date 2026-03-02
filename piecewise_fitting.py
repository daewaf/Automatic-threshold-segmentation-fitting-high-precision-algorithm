import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据
def load_data(file_path):
    data = np.loadtxt(file_path)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

# 分段拟合（基于自适应阈值的迭代式分段点识别）
def piecewise_fit(x, y):
    y_fit = np.zeros_like(y)
    n = len(x)
    
    # 识别分段点
    segment_points = [0]
    current_start = 0
    
    while current_start < n - 1:
        # 检查是否为水平直线段
        is_flat_line = False
        flat_end = current_start + 1
        
        # 检查当前点开始的连续点是否接近水平
        if current_start + 5 < n:
            # 计算前几个点的y值标准差
            y_values = y[current_start:current_start + 5]
            std_dev = np.std(y_values)
            
            # 如果标准差非常小，认为是水平直线
            if std_dev < 1e-8:
                is_flat_line = True
                # 找到水平直线的结束点
                for j in range(current_start + 5, n):
                    if abs(y[j] - y[current_start]) > 1e-8:
                        break
                    flat_end = j
        
        if is_flat_line:
            # 对于水平直线，直接作为一个分段
            segment_points.append(flat_end)
            current_start = flat_end
            continue
        
        # 检查是否为直线段（斜率变化较小）
        is_straight_line = False
        straight_end = current_start + 1
        
        # 计算起始点与第二个点的斜率（作为基准斜率）
        base_slope = 0.0
        
        if current_start + 1 < n and x[current_start + 1] != x[current_start]:
            base_slope = (y[current_start + 1] - y[current_start]) / (x[current_start + 1] - x[current_start])
            
            # 检查后续点的斜率变化
            max_slope_diff = 0.0
            for j in range(current_start + 2, min(current_start + 10, n)):
                if x[j] != x[current_start]:
                    current_slope = (y[j] - y[current_start]) / (x[j] - x[current_start])
                    slope_diff = abs(base_slope - current_slope)
                    max_slope_diff = max(max_slope_diff, slope_diff)
            
            # 如果斜率变化小于阈值，尝试直线拟合
            if max_slope_diff < 0.000001:
                # 找到直线段的结束点
                for j in range(current_start + 2, n):
                    if x[j] != x[current_start]:
                        current_slope = (y[j] - y[current_start]) / (x[j] - x[current_start])
                        slope_diff = abs(base_slope - current_slope)
                        if slope_diff >= 0.000001:
                            break
                        straight_end = j
                
                # 尝试直线拟合并计算MSE
                segment_x = x[current_start:straight_end + 1]
                segment_y = y[current_start:straight_end + 1]
                
                if len(segment_x) >= 2:
                    coefficients = np.polyfit(segment_x, segment_y, 1)
                    polynomial = np.poly1d(coefficients)
                    y_pred = polynomial(segment_x)
                    mse = np.mean((segment_y - y_pred) ** 2)
                    
                    if mse < 0.000001:
                        is_straight_line = True
        
        if is_straight_line:
            # 对于直线段，直接作为一个分段
            segment_points.append(straight_end)
            current_start = straight_end
            continue
        
        # 对每个分段点，从阈值100开始，逐步缩小
        threshold = 100.0
        best_end = current_start + 1
        best_mse = float('inf')
        
        while threshold >= 1e-10:  # 防止阈值过小
            # 使用当前阈值寻找分段点
            for i in range(current_start + 2, n):  # 从第三个点开始检查
                if x[i] != x[current_start]:  # 避免除零
                    # 计算起始点与当前点的斜率
                    current_slope = (y[i] - y[current_start]) / (x[i] - x[current_start])
                    # 计算斜率差值的绝对值
                    slope_diff = abs(base_slope - current_slope)
                    
                    # 对于水平直线，需要更严格的判断
                    if is_flat_line:
                        # 水平直线的斜率差异应该非常小
                        if abs(current_slope) > 1e-5:  # 只有当斜率明显偏离0时才认为是分段点
                            # 计算该分段的MSE
                            segment_x = x[current_start:i+1]
                            segment_y = y[current_start:i+1]
                            
                            if len(segment_x) >= 4:
                                coefficients = np.polyfit(segment_x, segment_y, 3)
                                polynomial = np.poly1d(coefficients)
                                y_pred = polynomial(segment_x)
                            else:
                                coefficients = np.polyfit(segment_x, segment_y, 1)
                                polynomial = np.poly1d(coefficients)
                                y_pred = polynomial(segment_x)
                            
                            mse = np.mean((segment_y - y_pred) ** 2)
                            
                            if mse < best_mse:
                                best_mse = mse
                                best_end = i
                                
                            # 如果MSE满足要求，停止寻找
                            if mse <= 0.000001:
                                break
                    else:
                        # 正常情况
                        if slope_diff > threshold:
                            # 计算该分段的MSE
                            segment_x = x[current_start:i+1]
                            segment_y = y[current_start:i+1]
                            
                            if len(segment_x) >= 4:
                                coefficients = np.polyfit(segment_x, segment_y, 3)
                                polynomial = np.poly1d(coefficients)
                                y_pred = polynomial(segment_x)
                            else:
                                coefficients = np.polyfit(segment_x, segment_y, 1)
                                polynomial = np.poly1d(coefficients)
                                y_pred = polynomial(segment_x)
                            
                            mse = np.mean((segment_y - y_pred) ** 2)
                            
                            if mse < best_mse:
                                best_mse = mse
                                best_end = i
                                
                            # 如果MSE满足要求，停止寻找
                            if mse <= 0.000001:
                                break
            
            # 如果当前阈值下的MSE满足要求，停止调整阈值
            if best_mse <= 0.000001:
                break
            
            # 否则缩小阈值10倍
            threshold /= 10
        
        # 添加找到的最佳分段点
        segment_points.append(best_end)
        current_start = best_end
    
    # 添加最后一个点
    if segment_points[-1] != n - 1:
        segment_points.append(n - 1)
    
    # 对每个分段进行最终拟合
    for i in range(len(segment_points) - 1):
        start = segment_points[i]
        end = segment_points[i + 1] + 1  # 包含结束点
        
        segment_x = x[start:end]
        segment_y = y[start:end]
        
        # 对每个段使用3次多项式拟合
        if len(segment_x) >= 4:  # 至少需要4个点来拟合3次多项式
            coefficients = np.polyfit(segment_x, segment_y, 3)
            polynomial = np.poly1d(coefficients)
            y_fit[start:end] = polynomial(segment_x)
        else:
            # 如果点数不足，使用线性拟合
            coefficients = np.polyfit(segment_x, segment_y, 1)
            polynomial = np.poly1d(coefficients)
            y_fit[start:end] = polynomial(segment_x)
    
    return y_fit, segment_points

# 计算均方误差
def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 单元测试函数
def test_slope_calculation():
    """测试斜率计算逻辑的正确性"""
    # 测试场景1：线性增长数据
    x1 = np.array([0, 1, 2, 3, 4, 5])
    y1 = np.array([0, 1, 2, 3, 4, 5])
    # 起始点(0,0)，第二个点(1,1)，斜率为1
    # 所有点的斜率都应该是1，差值为0
    
    # 测试场景2：斜率突变数据
    x2 = np.array([0, 1, 2, 3, 4, 5])
    y2 = np.array([0, 1, 2, 5, 8, 11])
    # 起始点(0,0)，第二个点(1,1)，斜率为1
    # 第四个点(3,5)，斜率为5/3 ≈ 1.666，差值约为0.666
    
    # 测试场景3：水平直线
    x3 = np.array([0, 1, 2, 3, 4, 5])
    y3 = np.array([2, 2, 2, 2, 2, 2])
    # 所有斜率都是0，差值为0
    
    print("=== 斜率计算单元测试 ===")
    
    # 测试场景1
    base_slope1 = (y1[1] - y1[0]) / (x1[1] - x1[0])
    for i in range(2, len(x1)):
        current_slope = (y1[i] - y1[0]) / (x1[i] - x1[0])
        slope_diff = abs(base_slope1 - current_slope)
        print(f"场景1 - 点{i}: 斜率={current_slope:.4f}, 差值={slope_diff:.4f}")
    
    # 测试场景2
    base_slope2 = (y2[1] - y2[0]) / (x2[1] - x2[0])
    for i in range(2, len(x2)):
        current_slope = (y2[i] - y2[0]) / (x2[i] - x2[0])
        slope_diff = abs(base_slope2 - current_slope)
        print(f"场景2 - 点{i}: 斜率={current_slope:.4f}, 差值={slope_diff:.4f}")
    
    # 测试场景3
    if x3[1] != x3[0]:
        base_slope3 = (y3[1] - y3[0]) / (x3[1] - x3[0])
        for i in range(2, len(x3)):
            current_slope = (y3[i] - y3[0]) / (x3[i] - x3[0])
            slope_diff = abs(base_slope3 - current_slope)
            print(f"场景3 - 点{i}: 斜率={current_slope:.4f}, 差值={slope_diff:.4f}")
    
    print("=== 测试完成 ===")

# 主函数
def main():
    # 加载数据
    file_path = 'c:\\Ai\\trae\\拟合算法\\ecg_data.txt'
    x, y = load_data(file_path)
    
    # 分段拟合
    y_piecewise, segment_points = piecewise_fit(x, y)
    mse_piecewise = calculate_mse(y, y_piecewise)
    
    # 打印结果
    print(f'分段点数量: {len(segment_points)}')
    print(f'分段数量: {len(segment_points) - 1}')
    print(f'分段点位置: {segment_points}')
    print(f'分段拟合 MSE: {mse_piecewise:.6f}')
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', label='原始数据')
    plt.plot(x, y_piecewise, 'y-', label=f'分段拟合 (MSE: {mse_piecewise:.6f})')
    
    # 标记分段点
    for point in segment_points:
        plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
    
    plt.title('ECG 分段拟合结果')
    plt.xlabel('时间')
    plt.ylabel('信号值')
    plt.legend()
    plt.grid(True)
    plt.savefig('piecewise_fitting_result.png')
    plt.show()

if __name__ == '__main__':
    # 运行单元测试
    test_slope_calculation()
    # 运行主函数
    main()
