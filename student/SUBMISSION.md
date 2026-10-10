# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Dương Dương
- MSSV: 2A202602498
- Email: duongduong160604@gmail.com
- Link repo (fork): https://github.com/duongduong1606/K4-L2L3-DAY23-DuongDuong-2A202602498-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): a6a06b5044d506704949830d5186c6ee456dafb1

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, [0, 198], training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: precision = 0.9701 (97.01%), recall = 0.7004 (70.04%), tp = 519, fp = 16, fn = 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse = 0.1503 m, matches = 502, sum_sq_err = 11.3369 m², ghost_track_frames = 0, missed_gt_frames = 239, mean_confirmed_tracks = 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse = 0.1358 m, matches = 502, sum_sq_err = 9.2642 m², ghost_track_frames = 0, missed_gt_frames = 239, mean_confirmed_tracks = 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Ở cả hai chế độ `lidar` và `fused`, số lượng matches (502), ghost track frames (0), missed GT frames (239) và mean confirmed tracks (2.5226) hoàn toàn trùng khớp nhau. Điều này phản ánh chính xác thiết kế **track-then-fuse**: chỉ có cảm biến LiDAR quyết định vòng đời của track (khởi tạo track mới, tính điểm score tồn tại, xác nhận và xóa track). Camera chỉ đóng vai trò cập nhật tinh chỉnh trạng thái EKF (vị trí, vận tốc) khi xe nằm trong trường nhìn (FOV) của camera FRONT, không làm thay đổi số lượng track.
  Về độ chính xác bám vết (tracking accuracy), chế độ `fused` đạt RMSE = 0.1358 m, cải thiện rõ rệt so với chế độ chỉ LiDAR (RMSE = 0.1503 m, giảm ~0.0145 m sai số tương đương ~9.6%), đồng thời tổng bình phương sai số `sum_sq_err` giảm từ 11.3369 m² xuống 9.2642 m². Việc bổ sung phép đo 2D pixel từ camera giúp thu hẹp độ bất định phương ngang của bộ lọc EKF mà không làm tăng bất kỳ ghost track nào (`ghost_track_frames = 0`), hoàn toàn đạt yêu cầu Rubric (RMSE <= 0.45 m và `rmse_fused - rmse_lidar <= 0.05 m`).

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - **Vector đo $z$:** LiDAR đo trực tiếp vị trí 3D trong không gian xe $z_{lidar} = [x, y, z]^T \in \mathbb{R}^3$ (mét) từ tâm hộp phát hiện trên lưới BEV. Camera đo tọa độ pixel 2D trên mặt phẳng ảnh $z_{cam} = [u, v]^T \in \mathbb{R}^2$ (pixel) từ tâm hộp 2D FRONT.
   - **Ma trận hiệp phương sai nhiễu $R$:** LiDAR có $R_{lidar} = \text{diag}(\sigma_x^2, \sigma_y^2, \sigma_z^2)$ với $\sigma = 0.1\text{ m}$, mô hình đo $h(x)$ là tuyến tính ($H$ kích thước 3×6 dạng ma trận chiếu đơn thuần). Camera có $R_{cam} = \text{diag}(\sigma_u^2, \sigma_v^2)$ với $\sigma = 5.0\text{ pixels}$, mô hình đo $h(x)$ là phi tuyến (phép chiếu Pinhole phối cảnh) nên Jacobian $H$ (kích thước 2×6) phải tính đạo hàm phi tuyến theo quy tắc dây chuyền (chain rule).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Cổng gating Mahalanobis ($d^2 = \gamma^T S^{-1} \gamma \le \chi^2_{\alpha, d}$) lọc bỏ các cặp đo lường - track không khả thi về mặt thống kê xác suất trước khi thực hiện gán tham lam (Greedy).
   - Khoảng cách Mahalanobis chuẩn hóa sai lệch đo lường $\gamma$ theo ma trận hiệp phương sai sai số $S = H P H^T + R$, có tính đến cả độ bất định trạng thái của track ($P$) lẫn độ nhiễu cảm biến ($R$). Khác với khoảng cách Euclidean thuần túy (chỉ đo khoảng cách hình học), khi $P$ lớn theo một hướng (ví dụ độ bất định cao theo chiều dọc vận tốc), một điểm đo xa về mặt hình học vẫn có thể có khoảng cách Mahalanobis nhỏ và hợp lệ. Ngược lại, khi $P$ nhỏ (ước lượng track đã rất tin cậy), cổng gating Mahalanobis tự động thu hẹp lại để loại bỏ đo lường sai lệch (clutter/false positive) và tránh gán nhầm sang xe khác.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline của lab là **track-then-fuse**: Duy trì một danh sách track duy nhất qua thời gian. Trong mỗi frame, EKF dự báo trạng thái một lần (`KF.predict`), sau đó cập nhật với đo LiDAR (`AssocL`) để quản lý vòng đời (tạo/xác nhận/xóa track), rồi cập nhật tiếp với đo Camera (`AssocC`) để tinh chỉnh trạng thái mà không thay đổi vòng đời track.
   - Minh chứng trên log và mã nguồn `run_lab.py` (dòng 198–210):
     - Bước detection chạy độc lập trên BEV LiDAR trước khi tracking.
     - Sau `KF.predict(track)`, thực hiện `assoc.associate_and_update(manager, observations, KF, lidar_sensor)`.
     - Tiếp theo mới thực hiện cập nhật camera: `assoc.associate_and_update(manager, observations, KF, camera_sensor)`.
     - Trong log `grade_run.log`, detection counts (`det_tp = 519, det_fp = 16, det_fn = 222`) ở từng frame giống hệt nhau giữa 2 mode `lidar` và `fused`, chứng minh detector hoạt động độc lập và chỉ thực hiện fusion ở mức track.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Khi ngoại thông số (extrinsic) hoặc nội thông số (intrinsic) của camera bị lệch (miscalibration), phép chiếu phối cảnh $h(x_{pred})$ trên mặt phẳng ảnh sẽ bị lệch có hệ thống (systematic bias) so với tọa độ đo pixel thực tế $z$.
   - **Triệu chứng cụ thể:**
     1. Vector phần dư $\gamma = z - h(x)$ sẽ có kỳ vọng khác 0 $(\mathbb{E}[\gamma] \neq 0)$, mang một giá trị bias cố định lệch theo trục sai lệch (thay vì dao động ngẫu nhiên phân phối chuẩn quanh 0).
     2. Khoảng cách Mahalanobis $d^2 = \gamma^T S^{-1} \gamma$ sẽ tăng vọt và vượt ngưỡng $\chi^2$, khiến các đo lường camera hợp lệ bị cổng gating loại bỏ (không gán được đo lường camera).
     3. Nếu ngưỡng gating quá rộng khiến phép đo vẫn bị gán, EKF update sẽ kéo trạng thái track lệch khỏi vị trí thực tế, dẫn đến sai số vị trí RMSE tăng cao.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - `associate_and_update` cần `sensor` tường minh vì ngay cả khi frame không có đo lường nào (`meas_list` rỗng), hàm vẫn phải gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)`. Trình quản lý track cần biết lượt này là của cảm biến nào: nếu là lượt LiDAR, mọi track nằm trong trường nhìn LiDAR mà không ghép được đo sẽ bị tính là một miss (`score -= 1/window`) và kiểm tra điều kiện xóa track.
   - LiDAR quyết định score/init/delete vì LiDAR đo trực tiếp vị trí 3D với tầm nhìn $360^\circ$ bao quát xung quanh xe, có khả năng phát hiện và duy trì track 3D độc lập. Ngược lại, camera chỉ quan sát góc hẹp phía trước (FRONT) dưới dạng pixel 2D và bị thiếu thông tin chiều sâu trực tiếp (scale ambiguity); camera không thể độc lập khởi tạo hay xác nhận vị trí 3D một cách tin cậy. Do đó, camera chỉ được dùng để cập nhật tinh chỉnh trạng thái EKF khi đối tượng nằm trong camera FOV, không can thiệp vào vòng đời track nhằm tránh sinh ra các ghost track giả.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - **Khởi tạo (Birth):** Khi có đo lường LiDAR chưa được gán, track mới được khởi tạo với trạng thái `'initialized'` và `score = 1/window` ($1/6 \approx 0.167$).
   - **Xác nhận (Confirmation):** Mỗi lần track được gán với đo lường LiDAR (hit), điểm tăng thêm `score = min(1.0, score + 1/window)`. Nếu `score > confirmed_threshold` ($0.8$, tương đương ít nhất 5 hits liên tiếp), track được nâng cấp lên trạng thái `'confirmed'`. Nếu chưa đủ ngưỡng confirm, track chuyển sang `'tentative'`.
   - **Giữ trạng thái confirmed:** Khi track nằm trong FOV LiDAR nhưng không được gán đo (miss), điểm bị trừ `score = score - 1/window`. Tuy nhiên, nếu track đã ở trạng thái `'confirmed'`, track vẫn giữ nguyên trạng thái `'confirmed'`, không bị giáng cấp về `'tentative'` chỉ vì một miss.
   - **Điều kiện xóa (Deletion):** Track bị xóa khỏi hệ thống khi thỏa mãn **bất kỳ** điều kiện nào sau đây:
     1. Phương sai vị trí theo phương ngang vượt ngưỡng: $P_{xx} > \text{max\_P}$ hoặc $P_{yy} > \text{max\_P}$ (với $\text{max\_P} = 9.0\text{ m}^2$).
     2. Track đã confirmed nhưng điểm giảm xuống quá thấp do nhiều lần miss liên tiếp: `score < delete_threshold` ($0.6$).
     3. Track chưa confirmed (`initialized` hoặc `tentative`) có điểm `score <= 0.0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI Assistant
- Dùng cho phần nào (hàm, câu hỏi, debug): Hỗ trợ triển khai thuật toán EKF 6D trong `kalman.py`, mô hình camera Pinhole và kiểm tra FOV trong `camera_fusion.py`, khoảng cách Mahalanobis và greedy matching trong `association.py`, quản lý vòng đời track trong `track_management.py`, gỡ lỗi môi trường Miniconda trên Windows, chạy tự động pipeline đánh giá và điền báo cáo.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy toàn bộ 122 unit tests bằng pytest (`pytest student/tests`), chạy pipeline đánh giá tích hợp trên toàn bộ 199 frames của tập dữ liệu Waymo (`fusion-run-lab --fusion compare --seed 0`), kiểm tra tính nhất quán giữa `metrics.json` và `grade_run.log`, và chạy script xác thực `python tools/check_submission.py`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
