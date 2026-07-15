# Học Biểu Diễn Đồ Thị Đa Quan Hệ cho Dự Đoán Kết Quả Kiểm Tra An Toàn Thực Phẩm tại Nhà Hàng

## 1. Bối cảnh và động lực nghiên cứu

An toàn thực phẩm là vấn đề y tế công cộng quan trọng. Các cơ quan quản lý định kỳ kiểm tra cơ sở kinh doanh thực phẩm, nhưng nguồn lực thanh tra thường hạn chế so với số lượng cơ sở cần giám sát. Một mô hình ước lượng nguy cơ cơ sở **không đạt** ở lần kiểm tra kế tiếp có thể hỗ trợ ưu tiên lịch thanh tra.

Hầu hết tiếp cận hiện có xem mỗi cơ sở như quan sát độc lập trong bài toán phân loại dạng bảng, bỏ qua các liên hệ thực tế như:

- lân cận địa lý;
- cùng loại hình kinh doanh;
- cùng khu vực hành chính (mã bưu chính);
- tương đồng lịch sử vi phạm.

Giả thuyết trung tâm: **thông tin lân cận trên đồ thị đa quan hệ có thể bổ sung tín hiệu rủi ro vượt ra ngoài đặc trưng lịch sử nội tại của từng cơ sở**. Nghiên cứu xây dựng khung học máy trên đồ thị, học embedding bằng GraphSAGE không giám sát, rồi dùng embedding cho mô hình phân loại nhằm dự đoán xác suất Fail ở lần kiểm tra kế tiếp.

> Phạm vi sử dụng: hỗ trợ **ưu tiên thanh tra**, không thay thế quyết định hành chính hay kết luận pháp lý.

---

## 2. Mục tiêu và bài toán

### 2.1. Mục tiêu

1. Xây dựng dữ liệu sạch, nhất quán theo thời gian cho bài toán dự đoán kết quả kiểm tra tiếp theo.
2. Mô hình hóa mạng lưới cơ sở dưới dạng **đồ thị đa quan hệ**.
3. Học biểu diễn nút bằng **GraphSAGE không giám sát** tại một mốc thời gian cố định.
4. Dùng embedding làm đặc trưng cho mô hình phân loại để ước lượng xác suất **Fail** trong lần kiểm tra đủ điều kiện kế tiếp.
5. Đánh giá theo tách mẫu thời gian, với chỉ số phù hợp dữ liệu mất cân bằng.

### 2.2. Phát biểu bài toán

Với mỗi lần kiểm tra “neo” (anchor) của một cơ sở tại thời điểm \(t\), mô hình dự đoán nhãn của **lần kiểm tra đủ điều kiện gần nhất sau \(t\)** trong chân trời \(H = 365\) ngày:

\[
y = \mathbb{1}\{\text{kết quả lần kiểm tra kế tiếp} = \text{Fail}\}.
\]

Đầu vào chỉ được phép dùng thông tin **đã quan sát trước \(t\)**. Kết quả, vi phạm và nội dung của lần kiểm tra mục tiêu bị loại khỏi đặc trưng.

---

## 3. Sơ đồ lưu trình nghiên cứu

```text
┌──────────────────────────┐
│ 1. Dữ liệu thanh tra thô │
│    (2010–2019)           │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 2. Tiền xử lý            │
│  - làm sạch, khử trùng   │
│  - liên kết danh tính    │
│  - định nghĩa nhãn 365d  │
│  - chống rò rỉ thời gian │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 3. EDA & hợp đồng dự đoán│
│  - mất cân bằng lớp      │
│  - thiếu dữ liệu         │
│  - biến thiên theo năm   │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 4. Đồ thị đa quan hệ     │
│  Nút: cơ sở (as-of time) │
│  Cạnh: geo / facility /  │
│        zip / history     │
│  Snapshot: 31/12/2016    │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 5. GraphSAGE không giám  │
│    sát → embedding 64-d  │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 6. Phân loại hạ nguồn    │
│  - XGBoost, MLP          │
│  - split theo thời gian  │
│  - chọn ngưỡng trên val  │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│ 7. Đánh giá & diễn giải  │
│  ROC-AUC, PR-AUC, F1,    │
│  Precision/Recall, CM    │
└──────────────────────────┘
```

Luồng trên nhấn mạnh hai lớp tách biệt: **biểu diễn cấu trúc** (bước 4–5) và **ra quyết định có giám sát** (bước 6–7). Mọi bước trước thời điểm dự đoán đều bị ràng buộc bởi nguyên tắc chỉ nhìn về quá khứ.

---

## 4. Nguồn dữ liệu

Bộ dữ liệu gồm hồ sơ thanh tra theo thời gian với các trường chính: định danh cơ sở, loại hình, mức rủi ro, ngày/loại/kết quả kiểm tra, mô tả vi phạm, tọa độ.

### 4.1. Quy mô

| Hạng mục | Giá trị |
|---|---:|
| Bản ghi thô | 196.825 |
| Bản ghi sau chuẩn hóa | 196.676 |
| Số thực thể cơ sở | 37.950 |
| Khoảng thời gian | 04/01/2010 – 04/12/2019 |
| Nhãn “kiểm tra kế tiếp” đủ điều kiện | 113.226 |
| Nhãn dương (Fail) | 20.571 |
| Nhãn âm (Pass / Pass w/ Conditions) | 92.655 |
| Tỷ lệ dương | ≈ 18,2% |

### 4.2. Phân bố kết quả trên dữ liệu thô

| Kết quả | Số lượng |
|---|---:|
| Pass | 106.066 |
| Fail | 38.087 |
| Pass w/ Conditions | 27.448 |
| Out of Business | 16.919 |
| No Entry | 6.324 |
| Not Ready | 1.912 |
| Business Not Located | 69 |

Fail là lớp thiểu số. Các kết quả mang tính vận hành (không vào được, chưa sẵn sàng, đã đóng cửa) không được gộp máy móc vào nhãn rủi ro vệ sinh.

---

## 5. Tiền xử lý và hợp đồng dự đoán

### 5.1. Làm sạch và chuẩn hóa

1. **Chuẩn hóa thời gian** và kiểm tra lược đồ.
2. **Khử trùng lặp** đúng nghĩa; xung đột cùng mã kiểm tra nhưng khác nội dung không bị gộp im lặng.
3. **Liên kết danh tính cơ sở:**
   - ưu tiên: *giấy phép + địa chỉ + mã bưu chính*;
   - dự phòng: *tên giao dịch + địa chỉ + mã bưu chính*.
4. **Bảo toàn thông tin thô**: thiếu dữ liệu được ghi nhận; việc loại khỏi đặc trưng diễn ra ở tầng hợp đồng mô hình.

### 5.2. Định nghĩa nhãn

Với mỗi bản ghi neo:

- chỉ xét kiểm tra **sau ngày neo** (không dùng cùng ngày);
- trong 365 ngày, lấy lần kiểm tra **đủ điều kiện sớm nhất**;
- gán `1` nếu `Fail`; `0` nếu `Pass` / `Pass w/ Conditions`;
- loại khỏi tập có giám sát nếu không có lần kiểm tra đủ điều kiện, hoặc chỉ gặp kết quả loại trừ.

### 5.3. Chống rò rỉ thời gian

- Đặc trưng, cạnh đồ thị và thống kê chuẩn hóa chỉ dùng dữ liệu **strictly before** thời điểm dự đoán.
- Không dùng kết quả/vi phạm của lần kiểm tra mục tiêu.
- Tách mẫu theo thời gian, không xáo trộn ngẫu nhiên.
- Không dùng thuộc tính “mới nhất toàn cục” nếu có thể chứa thông tin sau thời điểm dự đoán.

---

## 6. Nhận định từ phân tích mô tả

- Tỷ lệ Fail theo năm neo dao động khoảng **16–20%**, nên cần **temporal split**.
- Thiếu dữ liệu đáng kể ở mô tả vi phạm, một phần tọa độ/loại hình → không phụ thuộc một kênh thông tin duy nhất.
- Accuracy đơn lẻ không đủ; cần ROC-AUC, PR-AUC, F1, Precision/Recall.
- Đồ thị đa quan hệ kỳ vọng bổ sung tín hiệu khi lịch sử nội tại thưa hoặc cơ sở mới.

---

## 7. Xây dựng đồ thị đa quan hệ

### 7.1. Nút và đặc trưng

Mỗi nút là một **thực thể cơ sở** quan sát được trước cutoff **31/12/2016**. Snapshot thực nghiệm giới hạn **8.000 nút** (lấy mẫu xác định) để ổn định huấn luyện.

Đặc trưng as-of cutoff gồm: số lần kiểm tra trước đó, số/tỷ lệ Fail lịch sử, số ngày từ lần kiểm tra gần nhất, mức rủi ro, loại hình, tọa độ (kèm cờ thiếu), vector tần suất mã vi phạm đã rút gọn. Tổng chiều đặc trưng nút: **200**.

### 7.2. Quan hệ

| Quan hệ | Ý nghĩa | Số cạnh | Bậc TB |
|---|---|---:|---:|
| Địa lý (`geo`) | Lân cận ~750 m, tối đa 8 láng giềng | 60.909 | 7,61 |
| Loại hình (`facility`) | Cùng loại hình, giới hạn bậc | 107.524 | 13,44 |
| Mã bưu chính (`zip`) | Cùng ZIP, giới hạn bậc | 123.332 | 15,42 |
| Lịch sử vi phạm (`history`) | Tương đồng hồ sơ vi phạm | 34.288 | 4,29 |

### 7.3. Snapshot thời gian

Không dựng một đồ thị trên toàn bộ 2010–2019 rồi chỉ che nhãn test. Cấu trúc và thuộc tính được **đóng băng theo cutoff**, mô phỏng điều kiện triển khai khi hệ thống chỉ được nhìn quá khứ.

---

## 8. Học biểu diễn bằng GraphSAGE không giám sát

Học embedding không giám sát giúp:

1. tách **biểu diễn cấu trúc–lịch sử** khỏi **quyết định phân loại**;
2. tái sử dụng cùng không gian biểu diễn cho nhiều thuật toán hạ nguồn.

GraphSAGE học hàm tổng hợp từ láng giềng theo từng loại quan hệ, rồi hợp nhất qua các tầng phi tuyến. Mục tiêu huấn luyện dựa trên giả định cấu trúc: nút có liên kết nên gần nhau; mẫu âm ngẫu nhiên nên xa nhau. Nhãn Fail/Pass **không** tham gia mất mát ở bước này.

| Thành phần | Thiết lập |
|---|---|
| Kiến trúc | GraphSAGE đa quan hệ, 2 tầng |
| Chiều embedding | 64 |
| Số nút | 8.000 |
| Cutoff | 31/12/2016 |
| Epoch | 20 |
| Mất mát | ≈ 0,888 → 0,685 |

Mất mát giảm ổn định cho thấy mô hình học được tín hiệu cấu trúc; nhưng mất mát này **không đồng nhất** với mục tiêu dự đoán Fail.

---

## 9. Mô hình dự đoán hạ nguồn

### 9.1. Đặc trưng và tách mẫu

Đầu vào chính: **embedding 64 chiều**. Nhãn: Fail ở lần kiểm tra đủ điều kiện kế tiếp trong 365 ngày.

| Tập | Mốc neo | Số quan sát | Tỷ lệ Fail |
|---|---|---:|---:|
| Huấn luyện | ≤ 31/12/2016 | 21.760 | 18,14% |
| Kiểm định | ≤ 31/12/2017 | 2.695 | 19,93% |
| Kiểm tra | ≤ 30/09/2018 | 1.364 | 19,06% |

### 9.2. Thuật toán và chỉ số

Hai mô hình so sánh trên cùng embedding và cùng split:

1. **XGBoost** — ưu tiên ban đầu cho dữ liệu mất cân bằng, có điều chỉnh trọng số lớp dương.
2. **MLP nhỏ** — bộ so sánh phi tuyến trên không gian embedding.

Ngưỡng quyết định được chọn trên validation theo F1 rồi khóa trước test. Tiêu chí chọn mô hình: **PR-AUC** trên validation. Chỉ số báo cáo: Accuracy, ROC-AUC, PR-AUC, F1, Precision, Recall, ma trận nhầm lẫn.

---

## 10. Kết quả thực nghiệm

### 10.1. Tập kiểm định

| Mô hình | Accuracy | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0,376 | 0,575 | 0,250 | 0,343 | 0,816 | 0,217 |
| MLP | 0,478 | **0,580** | **0,253** | **0,346** | 0,695 | 0,231 |

MLP nhỉnh hơn rất nhỏ theo PR-AUC và được chọn cho test. Khoảng cách nhỏ gợi ý tín hiệu bị giới hạn chủ yếu ở tầng biểu diễn, không phải ở lựa chọn thuật toán phân loại.

### 10.2. Tập kiểm tra (MLP)

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0,437 |
| ROC-AUC | 0,537 |
| PR-AUC | 0,224 |
| F1 | 0,308 |
| Precision | 0,201 |
| Recall | 0,658 |
| Ngưỡng (khóa từ val) | ≈ 0,121 |

### 10.3. Ma trận nhầm lẫn (test)

|  | Dự đoán không Fail | Dự đoán Fail |
|---|---:|---:|
| **Thực tế không Fail** | TN = 425 | FP = 679 |
| **Thực tế Fail** | FN = 89 | TP = 171 |

- Bắt được ≈ **65,8%** ca Fail thật (Recall).
- Chỉ ≈ **20,1%** cảnh báo Fail là đúng (Precision).
- PR-AUC ≈ 0,224 chỉ cao hơn không nhiều so với tỷ lệ dương nền ≈ 0,191.
- Accuracy thấp hơn chiến lược “luôn đoán lớp đa số” vì ngưỡng thiên về Recall: trong thanh tra, **bỏ sót rủi ro cao thường đắt hơn báo động giả**.

---

## 11. Vì sao phương pháp cho kết quả còn thấp?

Kết quả ROC-AUC ≈ 0,54 và PR-AUC ≈ 0,22 phản ánh **tín hiệu yếu nhưng có thật**, không phải mô hình “hỏng”. Các nguyên nhân chính:

### 11.1. Lệch mục tiêu giữa học biểu diễn và bài toán dự đoán

GraphSAGE được huấn luyện **không giám sát** theo giả định “nút liên kết thì giống nhau”. Giả định này tối ưu **gần gũi cấu trúc**, không tối ưu trực tiếp việc tách Fail/Pass. Hai cơ sở lân cận hoặc cùng loại hình có thể giống nhau về ngữ cảnh nhưng **không nhất thiết đồng rủi ro** ở lần kiểm tra kế tiếp. Do đó, embedding có thể trung thực với đồ thị nhưng vẫn ít thông tin phân biệt nhãn.

### 11.2. Embedding-only tạo trần hiệu năng thấp

Bước phân loại gần như chỉ nhìn vector 64 chiều. Các tín hiệu giám sát mạnh và dễ giải thích—tỷ lệ Fail quá khứ, mức rủi ro, tần suất thanh tra, độ “nóng” gần đây—không được đưa trực tiếp vào classifier. Khi XGBoost và MLP cho kết quả gần nhau, điều đó cho thấy **không gian đặc trưng đã bão hòa tín hiệu yếu**, việc đổi thuật toán phân loại khó tạo bước nhảy.

### 11.3. Một snapshot cố định cho nhiều thời điểm dự đoán

Embedding được học tại cutoff 31/12/2016 rồi tái sử dụng cho các anchor năm 2017–2018. Trong khi đó, trạng thái rủi ro của cơ sở thay đổi theo thời gian. Thiết lập này tiện cho thí nghiệm nhưng **pha trộn tính “đúng thời điểm”**: một số neo sau cutoff nhận biểu diễn đã cũ, làm suy giảm liên hệ nhân quả giữa đặc trưng và nhãn.

### 11.4. Giới hạn quy mô và độ bao phủ quan hệ

Snapshot 8.000/37.950 thực thể giúp huấn luyện ổn định nhưng có thể cắt đuôi phân bố, làm mất một phần quan hệ và nhóm cơ sở thưa. Đồng thời, các quan hệ facility/ZIP bị giới hạn bậc để tránh bùng nổ cạnh; điều này cần thiết về mặt tính toán nhưng cũng **làm loãng tín hiệu cộng đồng**.

### 11.5. Bản chất bài toán khó và nhiễu nhãn vận hành

- Fail chỉ ≈ 18–20%; tín hiệu lớp thiểu số vốn yếu.
- Kết quả kiểm tra chịu ảnh hưởng thanh tra viên, thời điểm, loại hình thanh tra, và yếu tố không quan sát được (thay đổi chủ, sửa chữa đột xuất, sự kiện ngắn hạn).
- Dữ liệu chỉ có độ phân giải theo ngày, nên quan hệ nhân quả trong cùng ngày bị loại.
- Một số trường quan trọng (vi phạm, tọa độ, loại hình) bị thiếu, làm suy yếu cả đặc trưng nút lẫn cạnh history/geo.

### 11.6. Giao thức đánh giá trung thực hơn random split

Temporal split + chống rò rỉ thời gian **cố ý khó hơn** xáo trộn ngẫu nhiên. Nhiều công bố dùng random split dễ cho điểm cao hơn nhưng kém sát thực tế triển khai. Kết quả khiêm tốn một phần phản ánh việc đo lường chặt, không chỉ chất lượng mô hình.

### 11.7. Hệ quả tổng hợp

Các yếu tố trên cộng hưởng theo chuỗi:

```text
Mục tiêu không giám sát ≠ mục tiêu Fail
        ↓
Embedding ít tách lớp
        ↓
Classifier chỉ thấy tín hiệu yếu
        ↓
ROC-AUC/PR-AUC chỉ nhỉnh baseline
```

Do đó, hướng cải thiện then chốt không phải “đổi XGBoost bằng một classifier khác trong cùng embedding”, mà là **làm giàu đặc trưng as-of-time, bám mục tiêu hơn ở tầng biểu diễn, và đồng bộ snapshot với thời điểm dự đoán**.

---

## 12. Thảo luận ngắn

### 12.1. Đã đạt được

- Hợp đồng bài toán rõ: Fail ở lần kiểm tra đủ điều kiện kế tiếp trong 365 ngày.
- Đồ thị đa quan hệ có ý nghĩa chuyên môn.
- Tách lớp biểu diễn và lớp quyết định.
- Đánh giá theo thời gian, có kiểm soát rò rỉ.

### 12.2. Ý nghĩa thực tiễn hiện tại

Mô hình **chưa đủ vững để tự động hóa lịch thanh tra**. Giá trị gần hơn là:

- điểm rủi ro mang tính xếp hạng tham khảo;
- phân tầng cơ sở cần theo dõi sát;
- baseline có kiểm soát rò rỉ cho các vòng sau.

Nếu thử nghiệm hiện trường, nên dùng Top-K theo ngân sách thanh tra và theo dõi Precision@K/Recall@K, kèm phản hồi từ thanh tra viên.

---

## 13. Hướng phát triển

1. **Kết hợp embedding + đặc trưng lịch sử dạng bảng** (prior fail rate, recency, risk, facility, tần suất thanh tra).
2. **Embedding theo nhiều cutoff / suy luận quy nạp tại từng neo**, không tái sử dụng một snapshot cố định.
3. **Học bán giám sát hoặc có giám sát trên đồ thị**, để biểu diễn bám mục tiêu Fail.
4. **Ablation quan hệ** và so sánh baseline đầy đủ (bảng thuần, Node2Vec, GCN, GraphSAGE có giám sát, GAT).
5. **Đánh giá theo ngân sách thanh tra** (Recall@K, Precision@K, lift) và phân tích hiệu chỉnh xác suất.
6. Phân tích ổn định theo loại hình, khu vực, cơ sở mới/cũ.

---

## 14. Kết luận

Nghiên cứu triển khai quy trình hoàn chỉnh từ dữ liệu thanh tra thô đến dự đoán rủi ro kiểm tra kế tiếp theo hướng học máy trên đồ thị đa quan hệ. Điểm nhấn phương pháp gồm: định nghĩa nhãn bám tác vụ vận hành, kiểm soát rò rỉ thời gian, học embedding GraphSAGE trên bốn quan hệ có ý nghĩa chuyên môn, và đánh giá hạ nguồn bằng tách mẫu thời gian với chỉ số phù hợp mất cân bằng.

Kết quả cho thấy biểu diễn đồ thị mang **tín hiệu yếu nhưng có thật** (test ROC-AUC ≈ 0,54; PR-AUC ≈ 0,22). Khoảng cách nhỏ giữa XGBoost và MLP củng cố nhận định: bước đột phá tiếp theo nằm ở **nâng cấp đặc trưng và giao thức snapshot/biểu diễn**, không phải ở việc thay classifier trong cùng không gian embedding. Khung hiện tại là nền tảng phương pháp phù hợp để tiếp tục nghiên cứu ưu tiên thanh tra dựa trên đồ thị, nhưng cần thêm các vòng cải tiến trước khi hướng tới hỗ trợ vận hành tin cậy hơn.

---

## Phụ lục A. Số liệu then chốt

| Hạng mục | Giá trị |
|---|---|
| Giai đoạn dữ liệu | 2010–2019 |
| Bản ghi sau chuẩn hóa | 196.676 |
| Số cơ sở | 37.950 |
| Nhãn đủ điều kiện | 113.226 |
| Tỷ lệ Fail | ≈ 18,2% |
| Chân trời nhãn | 365 ngày |
| Cutoff đồ thị | 31/12/2016 |
| Số nút snapshot | 8.000 |
| Chiều embedding | 64 |
| Quan hệ | geo, facility, zip, history |
| Tách mẫu | train ≤ 2016; val ≤ 2017; test ≤ 2018-09 |
| Mô hình chọn | MLP |
| Test ROC-AUC / PR-AUC / F1 | 0,537 / 0,224 / 0,308 |
| Test Recall / Precision | 0,658 / 0,201 |

## Phụ lục B. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| Anchor inspection | Lần kiểm tra neo làm thời điểm dự đoán |
| Next eligible inspection | Lần kiểm tra đủ điều kiện gần nhất sau neo trong 365 ngày |
| Fail | Nhãn dương: không đạt |
| Snapshot cutoff | Mốc đóng băng đặc trưng và cấu trúc đồ thị |
| Embedding | Vector biểu diễn của cơ sở sau học trên đồ thị |
| Temporal split | Tách train/val/test theo thời gian |
| PR-AUC | Diện tích Precision–Recall; phù hợp lớp thiểu số |
| Recall / Precision | Tỷ lệ bắt đúng Fail / tỷ lệ cảnh báo Fail đúng |

---

*Tài liệu phục vụ viết báo cáo học thuật; lược bỏ chi tiết cài đặt phần mềm để tập trung vào phương pháp–dữ liệu–kết quả–luận giải.*
