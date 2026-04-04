@echo off
cd /d "C:\Users\1212\Desktop\IMU"

rem 1) 训练：D:\IMU_data 下 1.txt,2.txt,... 随机 1 个作测试集，其余作训练集（流式）
pip install -r requirements.txt
python .\train_blend_stream.py --data_dir D:\IMU_data --N_limit -1 --seed 0 --M 12 --lam_kernel 0.001 --poly_lam 0.001 --T_start_c 55 --margin_c 3 --out_dir outputs_stream_full

rem 2) 评测：读取 train_test_split.json 中的测试集文件
set N_used_eval=300000
python .\eval_blend.py --split_meta outputs_stream_full\train_test_split.json --N_used %N_used_eval% --blend_model_path outputs_stream_full\scheme_c_blend_model_stream.json --eps 5.0

rem 3) 画图：同样用测试集；x 轴为原始第 7 列温度
python .\tools\plot_blend_corrected.py --split_meta outputs_stream_full\train_test_split.json --N_used %N_used_eval% --blend_model_path outputs_stream_full\scheme_c_blend_model_stream.json --out_png outputs_stream_full\scheme_c_corrected_6d_vs_temp_blend_stream_raw.png --use_raw_temp_axis 1

echo [INFO] 全流程完成
pause

