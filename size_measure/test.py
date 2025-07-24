import cv2
import numpy as np

def detect_bottle_border(image_path, max_iter=10, tol=0.01, shrink_ratio=0.05):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    h, w = img.shape[:2]
    # 初始化 GrabCut mask
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    rect = (1, 1, w-2, h-2)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    # 第一次全图 GrabCut
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    prev_area = None
    final_mask = None

    for i in range(max_iter):
        # 前景二值化
        fg = np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')

        # 收敛判断
        area = cv2.countNonZero(fg)
        if prev_area is not None and abs(area - prev_area)/prev_area < tol:
            final_mask = fg
            print(f"✅ 第{i}轮收敛")
            break
        prev_area = area

        # 生成收缩框 refine mask
        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            final_mask = fg
            break
        c = max(cnts, key=cv2.contourArea)
        rect2 = cv2.minAreaRect(c)
        (cx, cy), (wr, hr), ang = rect2
        wr2 = max(wr * (1-shrink_ratio), 5)
        hr2 = max(hr * (1-shrink_ratio), 5)
        shrink_rect = ((cx, cy),(wr2, hr2), ang)

        # 重置 mask，用收缩矩形标记 probable FG
        mask[:] = cv2.GC_PR_BGD
        box = cv2.boxPoints(shrink_rect).astype(int)
        cv2.drawContours(mask, [box], -1, cv2.GC_PR_FGD, -1)

        # GrabCut 精炼
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

    if final_mask is None:
        final_mask = fg

    # 提取“边缘”＝形态学梯度
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    border = cv2.morphologyEx(final_mask, cv2.MORPH_GRADIENT, kernel)
    # 连通并封闭小断点
    border = cv2.morphologyEx(border, cv2.MORPH_CLOSE, kernel)

    # 找最大边缘轮廓
    cnts, _ = cv2.findContours(border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        print("❌ 未找到边缘轮廓")
        return
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    rect_final = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect_final).astype(int)

    # 绘制 & 保存
    out = img.copy()
    cv2.drawContours(out, [box], 0, (0,255,0), 2)
    cv2.imwrite("detected_image.jpg", out)

    rw, rh = rect_final[1]
    print(f"📏 最终边缘框 宽={rw:.2f}px, 高={rh:.2f}px, 面积={rw*rh:.2f}px²")
    print("💾 已保存到 detected_image.jpg")

# 调用
detect_bottle_border("small_05.jpg")
