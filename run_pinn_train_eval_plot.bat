@echo off
cd /d "C:\Users\1212\Desktop\IMU"

rem 1) 训练 PINN（从 YAML 配置读取全部参数，输出到 D:\IMU_output）
python train_pinn.py --config config\pinn_train.yaml

rem 2) 评测
python eval_pinn.py --model_path D:\IMU_output\pinn_model_best.pt --split_meta D:\IMU_output\train_test_split.json --N_used -1 --eps 5.0

rem 3) 画图
python tools\plot_pinn_results.py --model_path D:\IMU_output\pinn_model_best.pt --split_meta D:\IMU_output\train_test_split.json --N_used -1 --out_png D:\IMU_output\pinn_corrected_6d.png --show_components 1

rem 4) 启动 TensorBoard
echo [INFO] PINN 全流程完成
echo [INFO] 启动 TensorBoard: tensorboard --logdir D:\IMU_output\tb
pause
