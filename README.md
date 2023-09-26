# Vietcuna - Mô Hình Ngôn Ngữ Lớn Cho Người Việt

## Sử dụng Vietcuna
Vietcuna được tích hợp với HuggingFace
1) Khởi đầu bằng việc cài đặt Python và các lib cần thiết
```python
conda create -n 'vietcuna' python=3.9
conda activate vietcuna
pip install -r requirements.txt
```
2) Khởi động Vietcuna
```python
python launch.py
```
3) Sử dụng Vietcuna 7B với 4-bit (khuyên dùng để tiết kiệm VRAM)
```python
python launch.py --model_name vietcuna-7b --four-bit
``` 
## Roadmap
- [x] Vietcuna 3B
- [x] Vietcuna 7B - _đã phát hành bản alpha_
- [ ] Vietcuna 40B
- [x] UI Gradio
