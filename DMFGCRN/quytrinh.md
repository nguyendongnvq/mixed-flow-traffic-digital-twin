# Quy Trinh Xu Ly Du Lieu SUMO Va Train DMFGCRN

Tai lieu nay mo ta luong du lieu tu file SUMO ban dau den khi train va danh gia model DMFGCRN.

## 1. Du lieu dau vao tu SUMO

File goc:

```text
data/SumoData/forecast_baseline_node_level.csv
```

Schema chinh:

```text
begin,end,node_id,flow_count_5m,speed,e1_occupancy,e2_meanOccupancy,jam_veh
```

Y nghia cac cot dung cho model:

- `begin`: thoi diem bat dau interval, tinh bang giay.
- `end`: thoi diem ket thuc interval, tinh bang giay.
- `node_id`: id cua lane/node.
- `flow_count_5m`: so luong xe trong interval 5 phut.
- `e1_occupancy`: occupancy tu detector e1.
- `speed`: toc do trung binh.

Thong ke file hien tai:

```text
rows = 258840
timesteps = 8628
nodes = 30
interval = 300 seconds = 5 minutes
steps_per_day = 288
```

Ma tran du lieu day du:

```text
8628 timesteps * 30 nodes = 258840 rows
```

Khong co duplicate theo cap:

```text
(begin, node_id)
```

## 2. Process du lieu CSV thanh NPZ

Script xu ly:

```bash
python data/SumoData/process_sumo_data.py
```

Script nay thuc hien cac buoc:

1. Doc file `forecast_baseline_node_level.csv`.
2. Kiem tra day du cac cot bat buoc.
3. Kiem tra moi timestep co du 30 node.
4. Kiem tra moi node co du 8628 timestep.
5. Kiem tra khong co duplicate `(begin, node_id)`.
6. Sap xep `begin` tang dan va `node_id` theo thu tu lexicographic.
7. Tao cot `is_missing_speed` truoc khi fill missing.
8. Fill missing `speed = 0.0`.
9. Pivot tung feature thanh ma tran `(T, N)`.
10. Stack cac feature thanh tensor `(T, N, F)`.

Quy tac missing speed:

```text
is_missing_speed = 1.0 neu speed goc bi missing
is_missing_speed = 0.0 neu speed goc co gia tri
```

Sau do:

```text
speed = 0.0 voi cac dong speed goc bi missing
```

Ly do: tat ca cac dong missing `speed` hien tai deu co:

```text
flow_count_5m = 0
```

Nen `speed` duoc xem la undefined vi khong co xe di qua trong interval do. Cot `is_missing_speed` giup model phan biet `speed = 0` that va `speed = 0` do fill missing.

## 3. Output sau process

Script tao ra 2 file:

```text
data/SumoData/SUMO.npz
data/SumoData/SUMO_node_ids.csv
```

File `SUMO_node_ids.csv` luu thu tu node dung trong tensor.

File `SUMO.npz` co key chinh:

```text
data
```

Shape:

```text
SUMO.npz['data'].shape = (8628, 30, 4)
```

Feature order:

```text
0: flow_count_5m
1: e1_occupancy
2: speed
3: is_missing_speed
```

Luu y: `SUMO.npz` luu raw value, chua scale.

## 4. Cau hinh train SUMO

Config:

```text
config_file/SUMO_DMFGCRN.conf
```

Cac tham so chinh:

```ini
[data]
num_nodes = 30
lag = 12
horizon = 48
val_ratio = 0.2
test_ratio = 0.2
steps_per_day = 288

[model]
input_dim = 4
output_dim = 3

[train]
batch_size = 256
epochs = 50
lr_init = 0.003
early_stop = True
early_stop_patience = 15
real_value = True
```

Y nghia:

- `lag = 12`: model nhin 12 buoc qua khu.
- `horizon = 48`: model du doan 48 buoc tuong lai.
- Du lieu 5 phut/buoc, nen:

```text
horizon 1  = 5 minutes
horizon 3  = 15 minutes
horizon 6  = 30 minutes
horizon 12 = 60 minutes
horizon 24 = 120 minutes
horizon 36 = 180 minutes
horizon 48 = 240 minutes
```

## 5. Load dataset va them time features

Loader doc SUMO trong:

```text
lib/load_dataset.py
```

Voi dataset `SUMO`, loader doc:

```text
data/SumoData/SUMO.npz
```

Va giu nguyen 4 feature:

```text
[flow_count_5m, e1_occupancy, speed, is_missing_speed]
```

Sau do `lib/dataloader.py` tu them 2 time feature:

```text
time_in_day
day_in_week
```

Voi SUMO, `time_in_day` duoc tinh tu cot `begin` trong `SUMO.npz`, nen timestamp dau tien `begin = 3600` duoc hieu la 01:00 thay vi 00:00.

Sau khi them time features, input co 6 channel:

```text
0: flow_count_5m
1: e1_occupancy
2: speed
3: is_missing_speed
4: time_in_day
5: day_in_week
```

## 6. Tao window train/val/test

Tu raw data:

```text
data: (8628, 30, 4)
```

Voi:

```text
lag = 12
horizon = 48
```

So sample window:

```text
S = 8628 - 12 - 48 + 1 = 8569
```

Sau khi them time features:

```text
x: (8569, 12, 30, 6)
y: (8569, 48, 30, 6)
```

Split theo thu tu thoi gian:

```text
train = 60%
val   = 20%
test  = 20%
```

Voi config hien tai, shape xap xi:

```text
train x: (5142, 12, 30, 6)
val x:   (1714, 12, 30, 6)
test x:  (1713, 12, 30, 6)
```

Target dung de train/evaluate chi lay 3 feature dau:

```text
y[..., :3]
```

Shape target:

```text
(batch, 48, 30, 3)
```

## 7. Scaler va tranh leak test

Scaler chi fit tren train input:

```python
scaler = normalize_dataset(x_train[..., :continuous_dim], ...)
```

Voi SUMO:

```text
input_dim = 4
output_dim = 3
continuous_dim = 3
```

Nen scaler chi fit tren:

```text
flow_count_5m
e1_occupancy
speed
```

Sau do scaler duoc apply cho input cua ca train/val/test:

```text
x_train[..., :3]
x_val[..., :3]
x_test[..., :3]
```

Khong scale:

```text
is_missing_speed
time_in_day
day_in_week
```

Label `y` giu raw value. Khi tinh loss va metric, model output duoc inverse transform ve raw scale:

```python
output = scaler.inverse_transform(output)
loss = loss(output, label)
```

Ket luan ve leakage:

```text
Scaler fit bang train only.
Khong dung val/test de tinh mean/std.
Khong leak test vao train o buoc scaler.
```

## 8. Input va output cua model

Model nhan input:

```text
source[..., :4]
```

Bao gom:

```text
flow_count_5m
e1_occupancy
speed
is_missing_speed
```

Time embedding dung:

```text
source[..., 4] = time_in_day
source[..., 5] = day_in_week
```

Model predict 3 feature:

```text
flow_count_5m
e1_occupancy
speed
```

Output shape:

```text
(batch, 48, 30, 3)
```

`is_missing_speed` chi la input mask, khong phai target.

## 9. Lenh train

Chay train local hoac tren Colab:

```bash
python -u run.py --dataset SUMO --model DMFGCRN --device cuda:0
```

Neu train tren Colab, co the dung notebook:

```text
train_sumo_colab.ipynb
```

Notebook nay gom cac buoc:

1. Mount Google Drive.
2. Kiem tra GPU.
3. Verify `SUMO.npz`.
4. Ghi lai config SUMO.
5. Smoke test shape.
6. Chay train.
7. Doc metric summary.

## 10. Early stopping va checkpoint

Config co:

```ini
early_stop = True
early_stop_patience = 15
```

Nghia la train tu dung neu validation loss khong cai thien trong 15 epoch lien tiep.

Trainer luu:

```text
best_model.pth
best_test_model.pth
losses.xlsx
loss.png
horizon_metrics.csv
```

Duong dan output co dang:

```text
experiments/SUMO/<timestamp>/<seed>/
```

Vi du:

```text
experiments/SUMO/20260512110133/10/
```

## 11. Metric output

Sau test, trainer log metric theo tung horizon:

```text
MAE
RMSE
MAPE
sMAPE
```

File output:

```text
horizon_metrics.csv
```

Co cac cot:

```text
horizon,mae,rmse,mape,smape
```

Nhung horizon can quan tam:

```text
1, 3, 6, 12, 24, 36, 48
```

Y nghia theo thoi gian:

```text
1  -> 5 minutes
3  -> 15 minutes
6  -> 30 minutes
12 -> 60 minutes
24 -> 120 minutes
36 -> 180 minutes
48 -> 240 minutes
```

## 12. Tom tat pipeline

Toan bo pipeline:

```text
forecast_baseline_node_level.csv
        |
        v
process_sumo_data.py
        |
        v
SUMO.npz: (8628, 30, 4)
        |
        v
load_st_dataset('SUMO')
        |
        v
add time_in_day + day_in_week
        |
        v
x: (8569, 12, 30, 6)
y: (8569, 48, 30, 6)
        |
        v
split train/val/test
        |
        v
fit scaler on train x continuous features only
        |
        v
train DMFGCRN
        |
        v
predict: (batch, 48, 30, 3)
        |
        v
evaluate MAE/RMSE/MAPE/sMAPE by horizon
```

