# NewAnomalyDetector

Dự án mô phỏng giao thông bằng SUMO và phát hiện bốn loại bất thường:

- `accident`: tai nạn.
- `congestion_spike`: ùn tắc tăng đột biến.
- `zigzag`: phương tiện di chuyển zigzag.
- `sudden_stop`: phương tiện dừng đột ngột.

Hệ thống gồm ba phần chính:

1. Chạy mô phỏng SUMO và sinh nhãn bất thường.
2. Chuyển dữ liệu detector của SUMO thành dữ liệu node-level và các cửa sổ thời gian.
3. Huấn luyện, đánh giá và suy luận bằng LSTM Autoencoder hoặc DMFGCRN.

Luồng DMFGCRN đa nhãn là luồng nên dùng khi cần phát hiện tất cả loại bất thường
trong cùng một mô hình. Luồng LSTM được giữ lại để chạy các baseline đơn nhãn.

> Tất cả lệnh trong tài liệu này được chạy bằng PowerShell tại thư mục gốc của
> dự án. Ví dụ: `E:\Projects\NewAnomalyDetector`.

## 1. Cấu trúc thư mục

```text
NewAnomalyDetector/
|-- configs/                 Cấu hình huấn luyện và tạo dataset
|-- datasets/
|   |-- raw/sumo_data/       XML detector và nhãn thô từ SUMO
|   `-- processed/           CSV/NPZ đã xử lý
|-- DMFGCRN/                 Mã nguồn và checkpoint backbone DMFGCRN
|-- models/                  Checkpoint, metric, biểu đồ và kết quả suy luận
|-- scripts/                 Pipeline tự động và công cụ đánh giá
|-- src/
|   |-- evaluation/          Suy luận, hiệu chỉnh threshold và đánh giá
|   |-- models/              Các kiến trúc mô hình
|   |-- preprocessing/       Tạo cửa sổ dữ liệu
|   |-- process_rawdata/     Chuyển XML/CSV thành dữ liệu huấn luyện
|   |-- simulation/          Chạy SUMO và chèn bất thường
|   `-- train/               Huấn luyện mô hình
|-- sumo/                    Network, route, scenario và additional files
|-- main.py                  Entry point của baseline LSTM/DMFGCRN đơn nhãn
|-- requirements.txt         Dependency Python đã khóa phiên bản
`-- SUMO-Simulate.xlsx       Cấu hình lưu lượng đầu vào
```

## 2. Yêu cầu hệ thống

### 2.1 Phần mềm

- Windows 10/11 và PowerShell.
- Python 3.13 (phiên bản dùng để kiểm tra các dependency đã khóa). Môi trường `.venv` hiện tại của dự án đã được kiểm tra với
  Python 3.13.9.
- SUMO có cả `sumo.exe`, `sumo-gui.exe` và thư mục `tools`.
- Dung lượng trống đủ cho dataset và checkpoint trong `datasets/` và `models/`.

Mặc định các config huấn luyện DMFGCRN mới chạy trên CPU. Nếu đổi `device` thành
`cuda`, cần tự cài bản PyTorch tương thích với GPU/CUDA của máy.

### 2.2 Kiểm tra Python

```powershell
python --version
python -m pip --version
```

Nếu máy có nhiều phiên bản Python, có thể dùng Python Launcher:

```powershell
py -0p
```

Chỉ dùng cú pháp như `py -3.13` khi phiên bản đó xuất hiện trong kết quả `py -0p`.

### 2.3 Cài đặt và cấu hình SUMO

Sau khi cài SUMO, đặt `SUMO_HOME` trỏ đến thư mục cài đặt. Ví dụ:

```powershell
$env:SUMO_HOME = "E:\Projects\Sumo"
$env:Path = "$env:SUMO_HOME\bin;$env:Path"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"
```

Ba biến trên chỉ tồn tại trong cửa sổ PowerShell hiện tại. Để lưu cho tài khoản
Windows và dùng ở các terminal mở sau:

```powershell
[Environment]::SetEnvironmentVariable(
  "SUMO_HOME",
  "E:\Projects\Sumo",
  "User"
)

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable(
  "Path",
  "E:\Projects\Sumo\bin;$userPath",
  "User"
)

$userPythonPath = [Environment]::GetEnvironmentVariable("PYTHONPATH", "User")
[Environment]::SetEnvironmentVariable(
  "PYTHONPATH",
  "E:\Projects\Sumo\tools;$userPythonPath",
  "User"
)
```

Đóng và mở lại PowerShell, sau đó kiểm tra:

```powershell
sumo --version
sumo-gui --version
python -c "import traci; print(traci.__file__)"
```

`traci` được cung cấp bởi thư mục `tools` của SUMO, không nằm trong
`requirements.txt`. Nếu hai lệnh SUMO chạy được nhưng Python không import được
`traci`, kiểm tra lại `SUMO_HOME` và `PYTHONPATH`.

## 3. Cài đặt dự án

### 3.1 Mở thư mục dự án

```powershell
Set-Location E:\Projects\NewAnomalyDetector
```

### 3.2 Tạo môi trường ảo

```powershell
python -m venv .venv
```

Nếu đã kiểm tra Python Launcher có Python 3.13, có thể chọn rõ phiên bản:

```powershell
py -3.13 -m venv .venv
```

### 3.3 Kích hoạt môi trường

```powershell
.\.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script bằng lỗi `running scripts is disabled`, chỉ thay đổi
execution policy cho tiến trình hiện tại rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Sau khi kích hoạt, lệnh `python` phải trỏ vào `.venv`:

```powershell
python -c "import sys; print(sys.executable)"
```

### 3.4 Cài dependency

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` hiện khóa PyTorch bản CPU. Kiểm tra các thư viện chính:

```powershell
python -c "import numpy, pandas, sklearn, torch; print(torch.__version__)"
python -c "import traci; print('TraCI: OK')"
```

Kiểm tra các entry point mà không chạy huấn luyện:

```powershell
python main.py --help
python -m src.preprocessing.build_multilabel_dataset --help
python -m src.train.train_dmfgcrn_multilabel --help
python -m src.evaluation.infer_dmfgcrn_multilabel --help
```

## 4. Dữ liệu và checkpoint cần có

Bản GitHub giữ mã nguồn, cấu hình, network/route/detector SUMO, workbook đầu vào,
các bảng mapping tham chiếu và checkpoint backbone khoảng 9 MB. Dataset mô phỏng,
checkpoint huấn luyện mới, benchmark, báo cáo xuất tự động và archive không nằm
trong Git. Các file này vẫn được giữ trên máy khi bỏ theo dõi bằng Git.

### 4.1 Chạy với dữ liệu có sẵn

Để chạy nhanh mà không mô phỏng lại SUMO, kiểm tra các file sau:

```powershell
Test-Path datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s02.npz
Test-Path DMFGCRN\experiments\SUMO\20260512041404\10\best_model.pth
```

Sau khi clone, checkpoint phải có sẵn; dataset cần được tạo lại theo mục 8 hoặc
được cung cấp riêng. Khi đã chuẩn bị dữ liệu, cả hai lệnh nên trả về `True`:

- File `.npz` chứa các cửa sổ train/validation/test.
- File `.pth` là checkpoint backbone DMFGCRN dùng bởi mô hình đa nhãn.

Đường dẫn dataset, nguồn CSV, số node, feature và tỷ lệ chia dữ liệu được khai
báo trong file JSON tại `configs/`. Khi đổi dataset phải truyền cùng config đã
dùng để tạo dataset đó.

### 4.2 Tạo lại dữ liệu từ đầu

Cần có:

- `SUMO-Simulate.xlsx`.
- Các file network, route, scenario và additional trong `sumo/`.
- SUMO và TraCI đã được cấu hình theo mục 2.3.

Mô phỏng có thể chạy nhiều giờ tùy số ngày và cấu hình. Các pipeline sẽ tạo file
lớn và một số script từ chối ghi đè dataset đã tồn tại; nên đổi tên output hoặc
di chuyển artifact cũ trước khi chạy lại.

## 5. Chạy nhanh DMFGCRN đa nhãn

Luồng này dự đoán đồng thời `accident`, `congestion_spike`, `zigzag` và
`sudden_stop`. Một cửa sổ có thể chứa nhiều nhãn.

### 5.1 Tạo dataset NPZ từ CSV đã xử lý

Nếu file NPZ đã tồn tại và không cần tạo lại, bỏ qua bước này:

```powershell
python -m src.preprocessing.build_multilabel_dataset `
  --config configs\dmfgcrn_multilabel_flat_7d_s02.json `
  --output datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s02.npz
```

Có thể giới hạn số cửa sổ mỗi nguồn để smoke test:

```powershell
python -m src.preprocessing.build_multilabel_dataset `
  --config configs\dmfgcrn_multilabel_flat_7d_s02.json `
  --output datasets\processed\multilabel\dmfgcrn_multilabel_smoke.npz `
  --max-windows-per-source 100
```

### 5.2 Huấn luyện

Config `dmfgcrn_multilabel_flat_7d_s02.json` ở bước trước chỉ mô tả cách dựng
dataset. Khi huấn luyện, dùng `dmfgcrn_multilabel.json` vì file này có thêm
checkpoint backbone, hyperparameter và chính sách threshold; đường dẫn dataset
được ghi đè bằng `--dataset`.

Chạy đầy đủ:

```powershell
python -m src.train.train_dmfgcrn_multilabel `
  --config configs\dmfgcrn_multilabel.json `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s02.npz `
  --output-dir models\outputs_dmfgcrn_multilabel_flat_7d_s02
```

Smoke test ngắn để kiểm tra toàn bộ luồng trước khi train dài:

```powershell
python -m src.train.train_dmfgcrn_multilabel `
  --config configs\dmfgcrn_multilabel.json `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_smoke.npz `
  --output-dir models\outputs_dmfgcrn_multilabel_smoke `
  --head-epochs 1 `
  --fine-tune-epochs 1 `
  --max-train-samples 100 `
  --max-eval-samples 100
```

Artifact quan trọng trong thư mục output:

- `dmfgcrn_multilabel.pt`: checkpoint dùng cho suy luận.
- `test_metrics_by_class.csv`: precision, recall, F1 theo từng lớp.
- `test_summary.json`: tổng hợp kết quả trên tập test.
- Các file threshold và dự đoán được sinh theo cấu hình của lần train.

### 5.3 Suy luận

```powershell
python -m src.evaluation.infer_dmfgcrn_multilabel `
  --checkpoint models\outputs_dmfgcrn_multilabel_flat_7d_s02\dmfgcrn_multilabel.pt `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_flat_7d_s02.npz `
  --split test `
  --output-csv models\outputs_dmfgcrn_multilabel_flat_7d_s02\test_predictions.csv
```

Ý nghĩa cột `status` trong kết quả:

- `known_anomaly`: ít nhất một lớp vượt threshold riêng.
- `unknown_anomaly`: không lớp nào vượt threshold riêng nhưng residual tổng cao.
- `normal`: xác suất các lớp và residual đều dưới threshold.

## 6. Chạy mô hình DMFGCRN hai giai đoạn

Mô hình hai giai đoạn gồm detector bất thường và router phân loại loại bất
thường. Pipeline huấn luyện ba seed và đánh giá ensemble:

```powershell
python -m scripts.run_two_stage_training `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_multiscale_s02_s06_v2.npz `
  --config configs\dmfgcrn_two_stage_v2.json `
  --output-root models\outputs_dmfgcrn_two_stage_v2_s02_s06
```

Đánh giá ensemble:

```powershell
python -m scripts.evaluate_two_stage_ensemble `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_multiscale_s02_s06_v2.npz `
  --checkpoint-root models\outputs_dmfgcrn_two_stage_v2_s02_s06 `
  --output-dir models\outputs_dmfgcrn_two_stage_ensemble_v2_s02_s06
```

Smoke test có sẵn qua cờ `--smoke`:

```powershell
python -m scripts.run_two_stage_training `
  --dataset datasets\processed\multilabel\dmfgcrn_multilabel_windows_multiscale_s02_s06_v2.npz `
  --config configs\dmfgcrn_two_stage_v2.json `
  --output-root models\outputs_dmfgcrn_two_stage_smoke `
  --smoke
```

## 7. Chạy baseline LSTM/DMFGCRN đơn nhãn

`main.py` dùng cho baseline LSTM Autoencoder hoặc backend DMFGCRN đơn nhãn, tùy
giá trị `model_backend` trong config.

| Loại bất thường | Config LSTM |
|---|---|
| Tai nạn | `configs/lstm_autoencoder_accident.json` |
| Ùn tắc đột biến | `configs/lstm_autoencoder_congestion.json` |
| Zigzag | `configs/lstm_autoencoder_zigzag.json` |
| Dừng đột ngột | `configs/lstm_autoencoder_sudden_stop_target075.json` |

Chạy một config:

```powershell
python main.py --config configs\lstm_autoencoder_accident.json
```

Chạy lần lượt bốn config:

```powershell
$configs = @(
  "configs\lstm_autoencoder_accident.json",
  "configs\lstm_autoencoder_congestion.json",
  "configs\lstm_autoencoder_zigzag.json",
  "configs\lstm_autoencoder_sudden_stop_target075.json"
)

foreach ($config in $configs) {
  Write-Host "Đang chạy $config"
  python main.py --config $config
  if ($LASTEXITCODE -ne 0) {
    throw "Pipeline thất bại tại config: $config"
  }
}
```

Backend DMFGCRN đơn nhãn:

```powershell
python main.py --config configs\dmfgcrn_accident.json
```

Artifact thường có trong `output_dir` của config:

- `pilot_summary.json` và `pilot_summary.csv`.
- `event_eval_alert.csv`.
- `accident_window_result.csv`.
- Checkpoint `*.pt`.
- Các biểu đồ phân phối score, tỷ lệ phát hiện và độ trễ.

## 8. Tạo lại dữ liệu từ SUMO

### 8.1 Chạy thử có giao diện

Các config `mixed*` đi kèm đã có scenario và detector tương ứng. Khi tạo variant
mới, công cụ dùng template ngay trong repository:

```powershell
python -m src.simulation.build_mixed_simulation_config --variant mixed_smoke --days 1 --force
```

Lệnh `--force` ghi lại config/scenario của variant đã tồn tại; chỉ dùng khi muốn
tạo lại chúng. Để kiểm tra nhanh khởi động SUMO mà không chạy cả ngày, thêm
`--max-seconds 600 --no-gui` vào lệnh runner. Dữ liệu 600 giây chưa đủ để train.

Hai scenario cũ `named_roads_peak_hours.sumocfg` và
`named_roads_peak_hours_wide_coverage.sumocfg` thiếu route đầu vào tương ứng;
chúng đã được bỏ khỏi bản GitHub và chỉ giữ cục bộ. Route cũ
`named_roads.rou.xml` cũng được bỏ theo dõi vì không còn được tham chiếu.
Các flow runner dùng `named_roads_updated.*` trong hướng dẫn này không phụ thuộc
ba file đó. Preset một ngày `named_roads.sumocfg` vẫn được giữ vì có tham số riêng
và dùng route cập nhật. Các preset JSON và bộ SUMO tuần/seed khác nhau vẫn được giữ
để tái tạo thí nghiệm.


Dùng GUI để kiểm tra network, route và hành vi bất thường:

```powershell
python -m src.simulation.xlsx_traci_flow_runner `
  --config src\simulation\configs\xlsx_traci_flow_config.mixed_smoke.json
```

Chạy không giao diện để xử lý nhanh hơn:

```powershell
python -m src.simulation.xlsx_traci_flow_runner `
  --config src\simulation\configs\xlsx_traci_flow_config.mixed_smoke.json `
  --no-gui
```

Phải chạy runner bằng `python -m ...` vì module sử dụng relative import.

### 8.2 Pipeline mixed 7 ngày

Script sau tự động:

1. Tạo config và scenario SUMO.
2. Chạy mô phỏng.
3. Export detector XML thành timeseries.
4. Tạo dữ liệu forecast node-level.
5. Gắn nhãn theo node.
6. Đóng gói dataset DMFGCRN đa nhãn.

```powershell
.\scripts\run_mixed_flat_7d_pipeline.ps1 `
  -Variant mixed_flat_7d_s06 `
  -Days 7 `
  -TrafficScale 0.6
```

Các tùy chọn hữu ích:

- `-SkipSimulation`: bỏ qua mô phỏng nếu XML và nhãn đã tồn tại.
- `-ForceConfig`: ghi lại config/scenario do script tạo.
- `-MaxWindowsPerSource 100`: giới hạn dữ liệu để smoke test.

Log được ghi vào `.tmp\<variant>_run.log` và
`.tmp\<variant>_pipeline.log`.

### 8.3 Chạy thủ công từng bước

Ví dụ với variant `mixed_flat_7d_s02`:

```powershell
python -m src.simulation.xlsx_traci_flow_runner `
  --config src\simulation\configs\xlsx_traci_flow_config.mixed_flat_7d_s02.json `
  --no-gui
```

Export XML detector thành timeseries:

```powershell
python -m src.process_rawdata.export_timeseries `
  --modes mixed_flat_7d_s02 `
  --detector-dir datasets\raw\sumo_data\detectors `
  --scenarios-dir datasets\raw\sumo_data `
  --processed-root datasets\processed `
  --summary-path datasets\processed\mixed_flat_7d_s02\timeseries_export_summary.json `
  --strict
```

Tạo CSV forecast node-level:

```powershell
python -m src.process_rawdata.build_forecast_node_level_from_timeseries `
  --modes mixed_flat_7d_s02 `
  --processed-root datasets\processed `
  --suffix "" `
  --summary-path datasets\processed\mixed_flat_7d_s02\forecast_summary.json `
  --strict
```

Gắn nhãn đa lớp theo node:

```powershell
python -m src.process_rawdata.build_mixed_node_labels `
  --forecast datasets\processed\mixed_flat_7d_s02\forecast_mixed_flat_7d_s02_node_level.csv `
  --window-labels datasets\raw\sumo_data\anomaly_window_labels_mixed_flat_7d_s02.csv `
  --sensor-mapping datasets\processed\references\sensor_node_mapping.csv `
  --output datasets\processed\mixed_flat_7d_s02\forecast_mixed_flat_7d_s02_node_level_with_labels.csv
```

Sau đó chạy bước tạo NPZ ở mục 5.1 với config tương ứng.

## 9. Đọc kết quả đánh giá

Các metric chính:

- `precision`: tỷ lệ cảnh báo đúng trên tổng số cảnh báo.
- `recall`: tỷ lệ bất thường thực tế được phát hiện.
- `f1`: trung bình điều hòa giữa precision và recall.
- `normal_fp_rate`: tỷ lệ cửa sổ bình thường bị cảnh báo nhầm.
- `event_recall`: tỷ lệ sự kiện bất thường được phát hiện.
- `window_detect_ratio`: tỷ lệ cửa sổ bất thường được phát hiện.
- `mean_delay_seconds`: độ trễ phát hiện trung bình.
- `false_alerts_per_hour`: số cảnh báo sai mỗi giờ.

Không đánh giá mô hình chỉ bằng accuracy vì dữ liệu bất thường mất cân bằng.
Khi so sánh checkpoint, cần đọc đồng thời recall, F1 và false-positive rate trên
cùng dataset và cùng cách hiệu chỉnh threshold.

## 10. Xử lý lỗi thường gặp

### Không kích hoạt được `.venv`

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Hoặc không kích hoạt môi trường và gọi Python trực tiếp:

```powershell
.\.venv\Scripts\python.exe main.py --help
```

### `ModuleNotFoundError: No module named 'traci'`

```powershell
$env:SUMO_HOME = "E:\Projects\Sumo"
$env:PYTHONPATH = "$env:SUMO_HOME\tools;$env:PYTHONPATH"
python -c "import traci; print(traci.__file__)"
```

### Không tìm thấy `sumo` hoặc `sumo-gui`

```powershell
$env:Path = "$env:SUMO_HOME\bin;$env:Path"
Get-Command sumo
Get-Command sumo-gui
```

### Lỗi relative import khi chạy mô phỏng

Đúng:

```powershell
python -m src.simulation.xlsx_traci_flow_runner --config <config.json>
```

Sai:

```powershell
python src\simulation\xlsx_traci_flow_runner.py --config <config.json>
```

### Thiếu detector XML

Kiểm tra các file trong `datasets\raw\sumo_data\detectors`:

```powershell
Get-ChildItem datasets\raw\sumo_data\detectors\*.xml
```

Nếu thiếu, kiểm tra `sumocfg`, additional file, đường dẫn output detector và log
mô phỏng trong `.tmp`.

### Thiếu file nhãn

Với một variant mixed, cần có tối thiểu:

```text
anomaly_step_labels_<variant>.csv
anomaly_window_labels_<variant>.csv
anomaly_spawn_records_<variant>.json
```

Các file này phải nằm trong `datasets\raw\sumo_data` hoặc thư mục được truyền vào
`--scenarios-dir`.

### Không tìm thấy checkpoint DMFGCRN

Kiểm tra đường dẫn checkpoint trong config:

```powershell
Test-Path DMFGCRN\experiments\SUMO\20260512041404\10\best_model.pth
```

Nếu dùng checkpoint khác, cập nhật đúng trường checkpoint trong file JSON và
giữ nguyên kiến trúc, số node, input feature tương ứng với checkpoint đó.

### Hết RAM khi tạo dataset hoặc train

- Dùng `--max-windows-per-source` khi tạo NPZ.
- Dùng `--max-train-samples` và `--max-eval-samples` khi smoke test.
- Giảm `batch_size` trong config.
- Không nạp đồng thời nhiều dataset `.npz` lớn.

## 11. Quy tắc khi chạy lại thí nghiệm

- Không ghi đè checkpoint hoặc kết quả đang dùng; tạo `output-dir` mới.
- Lưu lại config cùng output để tái lập thí nghiệm.
- Giữ `seed`, cách chia train/validation/test và dataset cố định khi so sánh.
- Không chọn threshold trực tiếp trên tập test.
- Kiểm tra log và exit code trước khi dùng artifact của pipeline tiếp theo.

## 12. Kiểm tra trước khi đưa lên GitHub

```powershell
python -m pip check
python -m unittest discover -s tests -v
git status --short
git diff --cached --stat
```

Test `test_repository_inputs.py` kiểm tra đầu vào của toàn bộ config runner,
tham chiếu network/route/additional của mọi scenario được phát hành, checkpoint
trong config huấn luyện và bảng mapping để phát hiện file thiếu trên bản clone.
Unit test không thay thế chạy thử mô phỏng → xử lý → train → infer.

Các flow two-stage/ensemble cần checkpoint base và các seed được huấn luyện
trước bằng các script tương ứng; output thí nghiệm không được phát hành kèm Git.
`.gitignore` không xóa artifact trong lịch sử commit. Remote hiện tại có thể vẫn
trỏ GitLab; kiểm tra `git remote -v` trước khi tự cấu hình GitHub và push.
