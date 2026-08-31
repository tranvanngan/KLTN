# CLEF Project - Khóa luận tốt nghiệp

Dự án nghiên cứu và phát triển mã nguồn cho đề tài khóa luận tốt nghiệp của sinh viên ngành Kỹ thuật Phần mềm.
---
## 📂 Cấu trúc thư mục dự án

```text
CLEF/
│
├── CLEF_project/
│   ├── data/
│   │   └── raw/                  # Dữ liệu thô đầu vào (ví dụ: lab_temperature.csv, printer_power.csv)
│   ├── experiments/              # Các script chạy thử nghiệm, trực quan hóa và đánh giá
│   │   ├── figures/              # Các biểu đồ kết quả trực quan sinh ra từ thử nghiệm
│   │   ├── make_figures.py
│   │   ├── run_experiments.py
│   │   └── ...
│   ├── src/                      # Mã nguồn chính của hệ thống
│   │   ├── alignment.py
│   │   ├── check.py
│   │   ├── data_prep.py          # Tiền xử lý dữ liệu
│   │   ├── explainers.py         # Các mô hình giải thích mô hình (XAI: SHAP, LIME...)
│   │   ├── metrics.py            # Các hàm đo lường hiệu năng
│   │   ├── models.py             # Khởi tạo và huấn luyện mô hình
│   │   └── stats_tests.py        # Kiểm định thống kê
│   ├── requirements.txt          # Các thư viện Python cần thiết
│   └── pnginfo.js
│
└── .gitignore                    # Các file/thư mục bị bỏ qua bởi Git (bao gồm thư mục results/)
⚙️ Hướng dẫn cài đặt và chạy dự án
1. Yêu cầu hệ thống
Python 3.8 trở lên
Git
2. Cài đặt môi trường
Clone repository và di chuyển vào thư mục dự án:
Bash
git clone [https://github.com/tranvanngan/KLTN.git](https://github.com/tranvanngan/KLTN.git)
cd KLTN/CLEF_project
Cài đặt các thư viện phụ thuộc từ file requirements.txt:

Bash
pip install -r requirements.txt
3. Chạy thử nghiệm
Để chạy các kịch bản thử nghiệm chính của hệ thống, sử dụng lệnh:

Bash
python experiments/run_experiments.py
🛠️ Công nghệ sử dụng
Ngôn ngữ lập trình: Python
Thư viện chính: Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn
Công cụ hỗ trợ: Git, VS Code
👤 Tác giả
Họ và tên: Trần Văn Ngân
Ngành: Kỹ thuật Phần mềm
Trường: Đại học Kinh tế Thành phố Hồ Chí Minh (UEH)
---

git add README.md
git commit -m "Add detailed README.md"
git push origin main
