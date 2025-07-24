import cv2
import numpy as np

def detect_object_iterative_shrink(image_path, max_iter=10, tol=0.01, shrink_ratio=0.05):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    h, w = img.shape[:2]

    # 初始 GrabCut
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    rect = (1, 1, w-2, h-2)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    prev_area = None
    final_rect = None

    for i in range(max_iter):
        # 从 mask 提取前景二值图
        mask2 = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"⚠️ 第{i}轮无轮廓，停止迭代")
            break
        # 最大轮廓及凸包
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        minr = cv2.minAreaRect(hull)
        final_rect = minr
        rw, rh = minr[1]
        area = rw * rh

        # 收敛判断
        if prev_area and abs(area - prev_area)/prev_area < tol:
            print(f"✅ 第{i}轮收敛")
            break
        prev_area = area

        # 生成缩小的包围矩形作为下次 GrabCut 的 probable foreground
        (cx, cy), (w_rect, h_rect), ang = minr
        w2 = max(w_rect * (1 - shrink_ratio), 5)
        h2 = max(h_rect * (1 - shrink_ratio), 5)
        shrunk = ((cx, cy), (w2, h2), ang)

        # 重置 mask 并填充 probable FG 区域
        mask[:] = cv2.GC_PR_BGD
        box = cv2.boxPoints(shrunk).astype(int)
        cv2.drawContours(mask, [box], -1, cv2.GC_PR_FGD, -1)

        # GrabCut 精炼
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    # 输出最终结果
    if final_rect is not None:
        rw, rh = final_rect[1]
        area = rw * rh
        print(f"📏 最终（像素）宽={rw:.2f}, 高={rh:.2f}, 面积={area:.2f}")

        box = cv2.boxPoints(final_rect).astype(int)
        out = img.copy()
        cv2.drawContours(out, [box], 0, (0, 255, 0), 2)
        cv2.imwrite("detected_image.jpg", out)
        print("💾 已保存到 detected_image.jpg")
    else:
        print("❌ 未能获取有效轮廓")

# 运行
detect_object_iterative_shrink("small_05.jpg")
