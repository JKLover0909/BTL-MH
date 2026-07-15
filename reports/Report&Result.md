# Học Biểu Diễn Đồ Thị Đa Quan Hệ cho Dự Đoán Kết Quả Kiểm Tra An Toàn Thực Phẩm tại Nhà Hàng

## 1. Bối cảnh và động lực nghiên cứu

An toàn thực phẩm là một vấn đề y tế công cộng quan trọng. Các cơ quan quản lý địa phương định kỳ tiến hành kiểm tra cơ sở kinh doanh thực phẩm nhằm phát hiện vi phạm và giảm nguy cơ ngộ độc thực phẩm. Tuy nhiên, nguồn lực thanh tra thường bị hạn chế so với số lượng cơ sở cần giám sát. Trong bối cảnh đó, một mô hình dự đoán có khả năng ước lượng nguy cơ một cơ sở sẽ **không đạt** tại lần kiểm tra kế tiếp có thể hỗ trợ ưu tiên lịch thanh tra, tập trung vào các cơ sở có mức rủi ro cao hơn.

Hầu hết các tiếp cận hiện có xem mỗi cơ sở như một quan sát độc lập trong bài toán phân loại dạng bảng. Cách làm này bỏ qua các mối liên hệ thực tế giữa các cơ sở, ví dụ:

- các cơ sở nằm gần nhau về mặt địa lý;
- các cơ sở thuộc cùng loại hình kinh doanh;
- các cơ sở cùng khu vực hành chính (mã bưu chính);
- các cơ sở có lịch sử vi phạm tương đồng.

Giả thuyết trung tâm của nghiên cứu này là: **thông tin lân cận trên đồ thị đa quan hệ có thể bổ sung tín hiệu rủi ro vượt ra ngoài đặc trưng lịch sử nội tại của từng cơ sở**. Do đó, nghiên cứu xây dựng một khung học máy trên đồ thị, trong đó mỗi cơ sở là một nút, các quan hệ trên được mã hóa thành các loại cạnh khác nhau, và mô hình GraphSAGE không giám sát được dùng để học vector biểu diễn (embedding) của từng cơ sở. Các embedding này sau đó được đưa vào mô hình phân loại nhằm dự đoán xác suất thất bại ở lần kiểm tra kế tiếp.

> Lưu ý phạm vi sử dụng: kết quả nghiên cứu nhằm **hỗ trợ ưu tiên thanh tra**, không thay thế quyết định hành chính hay kết luận pháp lý của cơ quan quản lý.

---

## 2. Mục tiêu và bài toán nghiên cứu

### 2.1. Mục tiêu

1. Xây dựng tập dữ liệu sạch, nhất quán theo thời gian cho bài toán dự đoán kết quả kiểm tra tiếp theo.
2. Mô hình hóa mạng lưới cơ sở thực phẩm dưới dạng **đồ thị đa quan hệ**.
3. Học biểu diễn nút bằng **GraphSAGE không giám sát** trên đồ thị tại một mốc thời gian cố định.
4. Sử dụng embedding đã học làm đặc trưng đầu vào cho mô hình phân loại nhằm ước lượng xác suất cơ sở sẽ **Fail** trong lần kiểm tra đủ điều kiện kế tiếp.
5. Đánh giá hiệu năng theo giao thức tách mẫu theo thời gian, với các chỉ số phù hợp dữ liệu mất cân bằng.

### 2.2. Phát biểu bài toán

Với mỗi sự kiện kiểm tra “neo” (anchor inspection) của một cơ sở tại thời điểm \(t\), mô hình cần dự đoán nhãn của **lần kiểm tra đủ điều kiện gần nhất sau \(t\)** trong một chân trời thời gian cố định \(H = 365\) ngày:

\[
y = \mathbb{1}\{\text{kết quả lần kiểm tra kế tiếp} = \text{Fail}\}.
\]

Đặc trưng đầu vào tại thời điểm dự đoán chỉ được phép dựa trên thông tin **đã quan sát được trước \(t\)**. Mọi thông tin thuộc lần kiểm tra mục tiêu (kết quả, vi phạm, ghi chú, biện pháp khắc phục) đều bị loại khỏi đầu vào.

---

## 3. Nguồn dữ liệu

Nghiên cứu sử dụng bộ dữ liệu kiểm tra an toàn thực phẩm theo dạng hồ sơ thanh tra theo thời gian, với các trường chính gồm:

- định danh và thông tin định danh cơ sở (tên giao dịch, giấy phép, địa chỉ, mã bưu chính);
- loại hình cơ sở và mức rủi ro;
- ngày kiểm tra, loại kiểm tra, kết quả kiểm tra;
- mô tả vi phạm (nếu có);
- tọa độ địa lý.

### 3.1. Quy mô quan sát được

| Hạng mục | Giá trị |
|---|---:|
| Số bản ghi thô | 196.825 |
| Số bản ghi sau chuẩn hóa | 196.676 |
| Số thực thể cơ sở sau liên kết danh tính | 37.950 |
| Khoảng thời gian quan sát | 04/01/2010 – 04/12/2019 |
| Số nhãn “kiểm tra kế tiếp” đủ điều kiện | 113.226 |
| Số nhãn dương (Fail) | 20.571 |
| Số nhãn âm (Pass / Pass w/ Conditions) | 92.655 |
| Tỷ lệ dương trên tập nhãn đủ điều kiện | khoảng 18,2% |

### 3.2. Phân bố kết quả kiểm tra trên dữ liệu thô

| Kết quả | Số lượng |
|---|---:|
| Pass | 106.066 |
| Fail | 38.087 |
| Pass w/ Conditions | 27.448 |
| Out of Business | 16.919 |
| No Entry | 6.324 |
| Not Ready | 1.912 |
| Business Not Located | 69 |

Phân bố cho thấy lớp rủi ro cao (Fail) là thiểu số so với các kết quả đạt. Đồng thời tồn tại các kết quả mang tính vận hành (không vào được cơ sở, chưa sẵn sàng, đã đóng cửa) không nên được gộp máy móc vào nhãn dương/âm của bài toán dự đoán rủi ro vệ sinh.

---

## 4. Tiền xử lý và xây dựng hợp đồng dự đoán

### 4.1. Làm sạch và chuẩn hóa

Quy trình tiền xử lý gồm các bước chính:

1. **Kiểm tra lược đồ và tính hợp lệ thời gian.** Ngày kiểm tra được chuẩn hóa về dạng ngày-tháng-năm; các bản ghi không phân tích được thời gian bị coi là lỗi chất lượng dữ liệu.
2. **Xử lý trùng lặp.** Các dòng trùng khớp hoàn toàn được loại bỏ. Trường hợp cùng mã kiểm tra nhưng nội dung khác nhau được xem là xung đột và không được gộp im lặng.
3. **Liên kết danh tính cơ sở (entity resolution).** Một cơ sở không được xác định chỉ bằng số giấy phép, vì giấy phép có thể trống hoặc tái sử dụng. Khóa chính được xây dựng từ tổ hợp đã chuẩn hóa:

   - ưu tiên: *giấy phép + địa chỉ + mã bưu chính*;
   - dự phòng khi thiếu giấy phép: *tên giao dịch + địa chỉ + mã bưu chính*.

   Cách làm này cho phép theo dõi lịch sử theo “thực thể cơ sở–địa điểm” ổn định hơn so với việc dùng riêng giấy phép.

4. **Bảo toàn thông tin thô có giá trị phân tích.** Các trường thiếu (loại hình, tọa độ, mô tả vi phạm…) được ghi nhận trong báo cáo chất lượng, nhưng không bị xóa hàng loạt. Việc loại trừ khỏi đặc trưng diễn ra ở tầng hợp đồng mô hình, không phải bằng cách xóa vĩnh viễn khỏi dữ liệu nguồn đã chuẩn hóa.

### 4.2. Định nghĩa nhãn “lần kiểm tra kế tiếp”

Với mỗi bản ghi neo của một thực thể:

- xét các lần kiểm tra **sau ngày neo** (không dùng cùng ngày, do dữ liệu chỉ có độ phân giải theo ngày);
- trong cửa sổ 365 ngày, chọn lần kiểm tra **đủ điều kiện sớm nhất**;
- gán nhãn:
  - **1** nếu kết quả là `Fail`;
  - **0** nếu kết quả là `Pass` hoặc `Pass w/ Conditions`;
  - loại khỏi tập huấn luyện/đánh giá có giám sát nếu không tồn tại lần kiểm tra đủ điều kiện trong chân trời, hoặc chỉ quan sát được các kết quả loại trừ (`Out of Business`, `No Entry`, `Not Ready`, `Business Not Located`).

Cách định nghĩa này phản ánh mục tiêu vận hành: dự đoán rủi ro của **cuộc kiểm tra có thể diễn ra và có kết luận đạt/không đạt**, chứ không cố gán nhãn âm giả cho các trường hợp không quan sát được kết quả hợp lệ.

### 4.3. Nguyên tắc chống rò rỉ thông tin thời gian

Toàn bộ pipeline tuân thủ nguyên tắc *as-of-time*:

- mọi đặc trưng, quan hệ đồ thị và thống kê chuẩn hóa chỉ được tính từ dữ liệu có mốc thời gian **strictly before** thời điểm dự đoán;
- kết quả, vi phạm và nội dung của lần kiểm tra mục tiêu không được dùng làm đầu vào;
- không trộn ngẫu nhiên theo dòng; việc tách mẫu phải tôn trọng thứ tự thời gian;
- các thuộc tính “mới nhất toàn cục” của cơ sở (ví dụ loại hình hoặc tọa độ lấy từ bản ghi cuối cùng trong toàn bộ lịch sử) **không** được dùng trực tiếp làm đặc trưng lịch sử, vì có thể chứa thông tin sau thời điểm dự đoán.

---

## 5. Phân tích mô tả (EDA) — các nhận định chính

### 5.1. Mất cân bằng nhãn và biến thiên theo thời gian

Trên tập nhãn đủ điều kiện, tỷ lệ Fail dao động khoảng 16–20% theo năm neo. Tỷ lệ này không ổn định tuyệt đối theo thời gian, củng cố việc cần đánh giá theo lịch sử (temporal split) thay vì xáo trộn ngẫu nhiên.

### 5.2. Thiếu dữ liệu có cấu trúc

Một số trường quan trọng bị thiếu với tỷ lệ đáng kể, đặc biệt mô tả vi phạm và một phần tọa độ/loại hình. Do đó, mô hình không thể phụ thuộc hoàn toàn vào một kênh thông tin duy nhất; việc kết hợp lịch sử kết quả, quan hệ không gian và tương đồng vi phạm là cần thiết.

### 5.3. Hệ quả đối với thiết kế mô hình

- Bài toán mang tính **mất cân bằng lớp**, nên độ chính xác (Accuracy) không đủ để kết luận hiệu năng.
- Các chỉ số xếp hạng và nhạy với lớp thiểu số (ROC-AUC, PR-AUC, F1, Precision/Recall) cần được báo cáo đồng thời.
- Đồ thị đa quan hệ được kỳ vọng bổ sung tín hiệu khi lịch sử nội tại của cơ sở thưa hoặc mới xuất hiện.

---

## 6. Xây dựng đồ thị đa quan hệ

### 6.1. Nút (nodes)

Mỗi nút tương ứng một **thực thể cơ sở** còn quan sát được trước một mốc cắt thời gian (snapshot cutoff). Trong thực nghiệm chính, cutoff được chọn là **31/12/2016**, nhằm bảo đảm phần lớn giai đoạn huấn luyện diễn ra trước cửa sổ kiểm định và kiểm tra.

Để kiểm soát quy mô và ổn định huấn luyện biểu diễn, snapshot được giới hạn ở **8.000 nút** theo lấy mẫu xác định (deterministic), từ tập thực thể có lịch sử trước cutoff.

### 6.2. Đặc trưng nút tại thời điểm cắt

Với mỗi cơ sở tại cutoff, các đặc trưng lịch sử được tổng hợp chỉ từ các lần kiểm tra trước cutoff, bao gồm:

- số lần kiểm tra trước đó;
- số lần Fail trước đó và tỷ lệ Fail lịch sử;
- số ngày kể từ lần kiểm tra gần nhất;
- mức rủi ro và loại hình (dạng mã hóa);
- tọa độ (kèm cờ thiếu tọa độ);
- vector tần suất mã vi phạm lịch sử đã rút gọn chiều.

Tổng chiều đặc trưng nút sau mã hóa là **200**.

### 6.3. Các loại quan hệ (edges)

Đồ thị gồm bốn quan hệ, mỗi quan hệ có cơ chế giới hạn bậc để tránh bùng nổ cạnh:

| Quan hệ | Ý nghĩa chuyên môn | Số cạnh (snapshot) | Bậc trung bình xấp xỉ |
|---|---|---:|---:|
| Địa lý (`geo`) | Cơ sở lân cận trong bán kính khoảng 750 m, tối đa 8 láng giềng | 60.909 | 7,61 |
| Loại hình (`facility`) | Cơ sở cùng loại hình, liên kết ngang hàng có giới hạn | 107.524 | 13,44 |
| Mã bưu chính (`zip`) | Cơ sở cùng ZIP, liên kết ngang hàng có giới hạn | 123.332 | 15,42 |
| Lịch sử vi phạm (`history`) | Cơ sở có hồ sơ vi phạm tương đồng | 34.288 | 4,29 |

Thiết kế này phản ánh giả thuyết rằng rủi ro vệ sinh thực phẩm có thể đồng biến theo không gian, theo loại hình kinh doanh, theo khu vực, và theo “vân tay” vi phạm trong quá khứ.

### 6.4. Snapshot thời gian

Một nguyên tắc then chốt: **không xây một đồ thị duy nhất trên toàn bộ 2010–2019 rồi chỉ che nhãn tập test**. Thay vào đó, cấu trúc đồ thị và đặc trưng nút được đóng băng theo cutoff. Mọi cạnh và thuộc tính đều phải hợp lệ tại thời điểm cắt. Cách làm này mô phỏng điều kiện triển khai thực tế, khi hệ thống chỉ được phép nhìn về quá khứ.

---

## 7. Học biểu diễn bằng GraphSAGE không giám sát

### 7.1. Lý do chọn học không giám sát ở tầng đồ thị

Việc học embedding không giám sát có hai lợi ích nghiên cứu:

1. Tách bạch phần **biểu diễn cấu trúc–lịch sử** khỏi phần **ra quyết định phân loại**.
2. Cho phép tái sử dụng cùng một không gian biểu diễn cho nhiều thuật toán phân loại hạ nguồn và các phân tích ablations sau này.

### 7.2. Cơ chế học

GraphSAGE học hàm tổng hợp thông tin từ láng giềng thay vì học một vector cố định cho từng nút. Trên đồ thị đa quan hệ, thông tin được tổng hợp **theo từng loại cạnh**, sau đó hợp nhất với biểu diễn hiện tại của nút qua các tầng phi tuyến.

Mục tiêu huấn luyện không dùng nhãn Fail/Pass, mà dựa trên giả định cấu trúc:

- hai nút có liên kết (hoặc đồng xuất hiện trong quan hệ lân cận) nên có biểu diễn gần nhau;
- các nút lấy mẫu âm ngẫu nhiên nên có biểu diễn xa nhau.

### 7.3. Cấu hình thực nghiệm biểu diễn

| Thành phần | Thiết lập |
|---|---|
| Kiến trúc | GraphSAGE đa quan hệ, 2 tầng |
| Chiều embedding | 64 |
| Số nút snapshot | 8.000 |
| Cutoff | 31/12/2016 |
| Số vòng lặp huấn luyện | 20 |
| Hàm mất mát cuối | khoảng 0,685 (giảm từ khoảng 0,888) |
| Thiết bị tính toán | GPU CUDA |

Quá trình huấn luyện cho thấy hàm mất mát giảm ổn định theo epoch, cho thấy mô hình học được tín hiệu cấu trúc từ đồ thị. Embedding và trọng số encoder được lưu lại để phục vụ bước phân loại.

---

## 8. Mô hình dự đoán hạ nguồn

### 8.1. Đặc trưng đầu vào của bước phân loại

Ở thiết lập báo cáo này, đầu vào chính của mô hình phân loại là **vector embedding 64 chiều** đã học từ GraphSAGE. Nhãn là kết quả lần kiểm tra đủ điều kiện kế tiếp trong 365 ngày.

Cách thiết lập này đo lường trực tiếp giá trị dự đoán của biểu diễn đồ thị, nhưng cũng đặt ra giới hạn: nếu embedding chưa mã hóa đủ tín hiệu rủi ro có giám sát, hiệu năng phân loại sẽ bị trần bởi chất lượng biểu diễn.

### 8.2. Tách mẫu theo thời gian

| Tập | Điều kiện mốc neo | Số quan sát | Tỷ lệ Fail |
|---|---|---:|---:|
| Huấn luyện | đến 31/12/2016 | 21.760 | 18,14% |
| Kiểm định | đến 31/12/2017 | 2.695 | 19,93% |
| Kiểm tra | đến 30/09/2018 | 1.364 | 19,06% |

Chỉ các quan sát có cửa sổ nhãn 365 ngày đã “chín” so với phạm vi dữ liệu mới được giữ lại. Tỷ lệ dương tương đối ổn định giữa các tập, hỗ trợ so sánh công bằng.

### 8.3. Thuật toán phân loại được xem xét

Hai hướng phân loại được so sánh trên cùng đặc trưng embedding và cùng tách mẫu:

1. **XGBoost** — gradient boosting trên cây quyết định, có điều chỉnh trọng số lớp dương phù hợp dữ liệu mất cân bằng; đây là lựa chọn ưu tiên ban đầu cho dữ liệu dạng bảng/embedding dày.
2. **Mạng perceptron đa tầng (MLP) kích thước nhỏ** — bộ so sánh phi tuyến kiểu mạng nơ-ron trên không gian embedding.

Ngưỡng quyết định không giữ mặc định 0,5 một cách máy móc, mà được chọn trên tập kiểm định theo F1, rồi khóa lại trước khi đánh giá tập kiểm tra. Tiêu chí chọn mô hình trên kiểm định là **PR-AUC**.

### 8.4. Chỉ số đánh giá

Do lớp Fail chiếm khoảng một phần năm mẫu, nghiên cứu báo cáo đồng thời:

- **Accuracy**: tỷ lệ dự đoán đúng tổng quát, dễ hiểu nhưng dễ gây hiểu nhầm khi mất cân bằng;
- **ROC-AUC**: khả năng xếp hạng tổng quát giữa dương và âm;
- **PR-AUC**: mức độ hữu ích khi quan tâm lớp thiểu số (Fail);
- **F1, Precision, Recall**: hiệu năng tại một ngưỡng vận hành đã chọn;
- **Ma trận nhầm lẫn**: phân rã TN/FP/FN/TP để diễn giải chi phí bỏ sót và báo động giả.

---

## 9. Kết quả thực nghiệm

### 9.1. Kết quả học biểu diễn

Trên snapshot 8.000 nút, GraphSAGE không giám sát hội tụ theo hướng giảm mất mát qua 20 epoch (khoảng 0,888 → 0,685). Điều này cho thấy mô hình khai thác được cấu trúc đa quan hệ. Tuy nhiên, mất mát không giám sát **không đồng nhất** với mục tiêu dự đoán Fail; do đó cần bước đánh giá có giám sát riêng.

### 9.2. So sánh mô hình trên tập kiểm định

| Mô hình | Accuracy | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0,376 | 0,575 | 0,250 | 0,343 | 0,816 | 0,217 |
| MLP | 0,478 | **0,580** | **0,253** | **0,346** | 0,695 | 0,231 |

Theo PR-AUC trên kiểm định, **MLP nhỉnh hơn XGBoost ở mức rất nhỏ** và được chọn làm mô hình báo cáo trên tập kiểm tra. Khoảng cách giữa hai mô hình không lớn, gợi ý rằng với cùng embedding hiện tại, phần lớn tín hiệu đã bị giới hạn ở tầng biểu diễn chứ không phải ở lựa chọn thuật toán phân loại.

### 9.3. Hiệu năng trên tập kiểm tra (mô hình được chọn: MLP)

| Chỉ số | Giá trị |
|---|---:|
| Accuracy | 0,437 |
| ROC-AUC | 0,537 |
| PR-AUC | 0,224 |
| F1 | 0,308 |
| Precision | 0,201 |
| Recall | 0,658 |
| Ngưỡng quyết định (khóa từ kiểm định) | ≈ 0,121 |

### 9.4. Ma trận nhầm lẫn trên tập kiểm tra

|  | Dự đoán không Fail | Dự đoán Fail |
|---|---:|---:|
| **Thực tế không Fail** | TN = 425 | FP = 679 |
| **Thực tế Fail** | FN = 89 | TP = 171 |

Diễn giải vận hành:

- Mô hình bắt được khoảng **65,8%** các ca Fail thật (Recall).
- Tuy nhiên, trong các cảnh báo “nguy cơ Fail”, chỉ khoảng **20,1%** là đúng (Precision).
- Số báo động giả cao (FP = 679) cho thấy nếu dùng trực tiếp làm danh sách ưu tiên thanh tra, chi phí rà soát sẽ lớn.
- PR-AUC ≈ 0,224 chỉ cao hơn không nhiều so với tỷ lệ dương nền ≈ 0,191, cho thấy năng lực xếp hạng lớp thiểu số còn yếu.

### 9.5. Đọc kết quả trong bối cảnh mất cân bằng

Accuracy thấp hơn “luôn dự đoán lớp đa số” là hệ quả của việc chọn ngưỡng thiên về Recall. Điều này phù hợp mục tiêu thanh tra: **bỏ sót cơ sở rủi ro cao thường đắt hơn báo động giả**. Tuy nhiên, mức Precision hiện tại cho thấy biểu diễn chưa đủ tách bạch hai lớp để vừa giữ Recall cao vừa kiểm soát báo động giả.

---

## 10. Thảo luận

### 10.1. Những gì đã đạt được

1. **Hợp đồng bài toán rõ ràng:** dự đoán Fail ở lần kiểm tra đủ điều kiện kế tiếp trong 365 ngày, với nguyên tắc chỉ dùng thông tin quá khứ.
2. **Đồ thị đa quan hệ có ý nghĩa chuyên môn:** không gian, loại hình, khu vực, và tương đồng lịch sử vi phạm được mô hình hóa tường minh.
3. **Tách lớp biểu diễn và lớp quyết định:** GraphSAGE học embedding không giám sát; phân loại hạ nguồn có thể thay thế/so sánh độc lập.
4. **Giao thức đánh giá trung thực hơn random split:** train/validation/test theo thời gian, có xét điều kiện chín của nhãn.

### 10.2. Giới hạn chính

1. **Trần hiệu năng đến từ embedding không giám sát.** Biểu diễn được tối ưu cho gần gũi cấu trúc, không trực tiếp tối ưu khả năng tách Fail/Pass. Hệ quả là XGBoost và MLP cho kết quả gần nhau và chỉ vượt baseline yếu.
2. **Một snapshot embedding dùng chung cho nhiều mốc neo sau cutoff.** Cách làm này tiện cho thí nghiệm nhưng chưa phải thiết lập lý tưởng. Về nguyên tắc, mỗi thời điểm dự đoán nên có trạng thái đặc trưng/đồ thị as-of riêng, hoặc cơ chế suy luận quy nạp theo thời điểm.
3. **Giới hạn quy mô snapshot (8.000 nút)** giúp huấn luyện ổn định nhưng có thể làm mất một phần đuôi phân bố thực thể và quan hệ.
4. **Chưa tích hợp đầy đủ đặc trưng bảng lịch sử có giám sát** (ví dụ tương tác giữa tỷ lệ Fail quá khứ, tần suất thanh tra, thay đổi rủi ro) vào cùng mô hình với embedding. Đây khả năng là hướng cải thiện có tác động lớn hơn việc chỉ đổi thuật toán phân loại.
5. **Chưa có thực nghiệm ablation theo từng loại quan hệ** và chưa so sánh đầy đủ với các baseline đồ thị khác (GCN, GAT, Node2Vec) như định hướng ban đầu của đề cương.
6. **Độ phân giải thời gian theo ngày** buộc phải loại quan hệ nhân quả trong cùng ngày; một phần tín hiệu ngắn hạn có thể bị mất.

### 10.3. Ý nghĩa thực tiễn

Ở mức hiệu năng hiện tại, mô hình **chưa đủ vững để tự động hóa việc lập lịch thanh tra**. Giá trị thực tiễn gần hơn là:

- cung cấp một **điểm rủi ro mang tính xếp hạng tham khảo**;
- hỗ trợ phân tầng cơ sở thành nhóm cần theo dõi sát hơn;
- làm baseline có kiểm soát rò rỉ thời gian cho các vòng mô hình sau.

Nếu triển khai thử nghiệm hiện trường, cần kèm ngưỡng công suất (ví dụ chỉ chọn Top-K cơ sở mỗi kỳ), theo dõi Precision@K/Recall@K theo ngân sách thanh tra thực tế, và có vòng phản hồi từ thanh tra viên.

---

## 11. Hướng phát triển tiếp theo

Theo thứ tự ưu tiên học thuật và thực tiễn:

1. **Kết hợp embedding với đặc trưng lịch sử dạng bảng** (prior fail rate, recency, risk, facility, tần suất thanh tra) trong cùng mô hình phân loại.
2. **Sinh embedding theo nhiều cutoff thời gian** hoặc suy luận quy nạp tại từng thời điểm neo, thay vì tái sử dụng một snapshot cố định.
3. **Học bán giám sát / có giám sát trên đồ thị** với nhãn Fail chỉ từ giai đoạn huấn luyện, để biểu diễn bám sát mục tiêu dự đoán hơn mất mát cấu trúc thuần.
4. **Ablation quan hệ:** đo đóng góp riêng của geo, facility, ZIP, history.
5. **So sánh hệ thống baseline:** Logistic Regression và XGBoost trên đặc trưng bảng thuần; Node2Vec+MLP; GCN; GraphSAGE có giám sát; GAT.
6. **Đánh giá theo ngân sách thanh tra:** Recall@K, Precision@K, lift theo từng kỳ, và phân tích hiệu chỉnh xác suất (calibration).
7. **Phân tích công bằng và ổn định:** hiệu năng theo loại hình, theo khu vực, theo cơ sở mới/cũ; kiểm tra độ nhạy với chính sách liên kết danh tính.

---

## 12. Kết luận

Nghiên cứu đã triển khai một quy trình hoàn chỉnh từ dữ liệu thanh tra thô đến dự đoán rủi ro kiểm tra kế tiếp theo hướng học máy trên đồ thị đa quan hệ. Điểm nhấn phương pháp là:

- định nghĩa nhãn bám sát tác vụ “lần kiểm tra đủ điều kiện kế tiếp”;
- kiểm soát rò rỉ thời gian xuyên suốt tiền xử lý, dựng đồ thị và đánh giá;
- học biểu diễn nút bằng GraphSAGE không giám sát trên bốn quan hệ có ý nghĩa chuyên môn;
- đánh giá hạ nguồn bằng phân loại trên embedding với tách mẫu thời gian và chỉ số phù hợp mất cân bằng.

Kết quả định lượng cho thấy biểu diễn đồ thị hiện tại mang **tín hiệu yếu nhưng có thật** đối với nhiệm vụ dự đoán Fail (ROC-AUC kiểm tra ≈ 0,54; PR-AUC ≈ 0,22). Khoảng cách nhỏ giữa XGBoost và MLP củng cố nhận định rằng bước đột phá tiếp theo nằm ở **nâng cấp đặc trưng và giao thức snapshot/biểu diễn**, hơn là ở việc thay thế một thuật toán phân loại bằng một thuật toán khác trong cùng không gian embedding.

Tóm lại, khung đề xuất là một nền tảng phương pháp phù hợp để tiếp tục nghiên cứu ưu tiên thanh tra dựa trên đồ thị; tuy nhiên, để hướng tới hỗ trợ vận hành tin cậy hơn, cần các vòng cải tiến về đặc trưng as-of-time, học biểu diễn bám mục tiêu, và đánh giá theo ràng buộc nguồn lực thanh tra thực tế.

---

## Phụ lục A. Tóm tắt số liệu then chốt

| Hạng mục | Giá trị |
|---|---|
| Giai đoạn dữ liệu | 2010–2019 |
| Bản ghi sau chuẩn hóa | 196.676 |
| Số cơ sở | 37.950 |
| Nhãn đủ điều kiện | 113.226 |
| Tỷ lệ Fail (nhãn đủ điều kiện) | ≈ 18,2% |
| Chân trời nhãn | 365 ngày |
| Cutoff đồ thị | 31/12/2016 |
| Số nút snapshot | 8.000 |
| Chiều embedding | 64 |
| Quan hệ đồ thị | geo, facility, zip, history |
| Tách mẫu | train ≤ 2016; val ≤ 2017; test ≤ 2018-09 |
| Mô hình chọn trên validation | MLP |
| Test ROC-AUC | 0,537 |
| Test PR-AUC | 0,224 |
| Test F1 | 0,308 |
| Test Recall / Precision | 0,658 / 0,201 |

## Phụ lục B. Thuật ngữ dùng trong báo cáo

| Thuật ngữ | Nghĩa trong nghiên cứu |
|---|---|
| Anchor inspection | Lần kiểm tra neo dùng làm thời điểm dự đoán |
| Next eligible inspection | Lần kiểm tra đủ điều kiện gần nhất sau thời điểm neo trong chân trời 365 ngày |
| Fail | Nhãn dương: không đạt kiểm tra |
| Snapshot cutoff | Mốc thời gian đóng băng đặc trưng và cấu trúc đồ thị |
| Embedding | Vector biểu diễn số học của cơ sở sau khi học trên đồ thị |
| Temporal split | Tách huấn luyện/kiểm định/kiểm tra theo thời gian |
| PR-AUC | Diện tích dưới đường Precision–Recall; phù hợp lớp thiểu số |
| Recall | Tỷ lệ phát hiện đúng các ca Fail thật |
| Precision | Tỷ lệ cảnh báo Fail đúng trong mọi cảnh báo Fail |

---

*Tài liệu này mô tả quy trình chuyên môn và kết quả thực nghiệm phục vụ viết báo cáo học thuật. Các chi tiết cài đặt phần mềm, tổ chức mã nguồn và tham số kỹ thuật hệ thống được cố ý lược bỏ để tập trung vào nội dung phương pháp–dữ liệu–luận giải.*
