#### 项目实现功能

- 足迹定位检测与矫正
- 单张和对比矫正图的测量和标注
- 原图打印（纸张选择【A3，A4，A5】，打印比例）

#### 核心方法

- `YOLOv5`目标检测
- 仿射变换
- `pyqt5`的使用

#### 文件信息

- `funcclass`文件是图像测量窗口，包括单张图片的测量和保存；
- `contrastpic`文件在单张图片的基础上增加对比图像的测量和保存
- `footprint`文件是主窗口，包括矫正和测量分析及打印（一版本）
- `footprint_plus`文件是主窗口，在原始footprint基础上增加了
  - 标注图像对纸张和比例的选择
  - 对比图像的处理

使用python embeded的方式打包参见博客https://www.mulindya.com/2021/12/29/project_tool/usePyembeded/
